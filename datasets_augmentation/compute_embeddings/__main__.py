import copy
import logging
import math
import os
from argparse import ArgumentParser
from multiprocessing import cpu_count
from typing import Dict, Generator

import torch
from datasets import Dataset, Sequence, Value, concatenate_datasets
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichModelSummary
from lightning_fabric.utilities.distributed import _distributed_available as distributed_available
from lightning_fabric.utilities.rank_zero import rank_zero_info
from tqdm import tqdm

from datasets_augmentation.compute_embeddings.data import get_dataloader, prepare_dataset
from datasets_augmentation.compute_embeddings.model import EncodingModel
from datasets_augmentation.utilities import TQDMProgressBar, cache_files_reader, clean_folder, split_dataset_in_chunks
from transformers_framework.utilities.arguments import add_trainer_args, get_trainer_args_from_hyperparameters
from transformers_framework.utilities.classes import ExtendedNamespace


os.environ['TOKENIZERS_PARALLELISM'] = "true"
logging.getLogger("transformers").setLevel(logging.ERROR)  # too much complains of the tokenizers


OUTPUT_LOGGING = '/tmp/datasets_augmentation'


def encode(
    input_dataset: Dataset = None,
    model: EncodingModel = None,
    trainer: Trainer = None,
    input_field: str = None,
    encoding_batch_size: int = None,
    remove_stopwords: bool = None,
    max_encoding_length: bool = None,
    tmp_dir: str = None,
    chunk_id: int = 0,
) -> Generator[Dict, None, None]:
    r""" Process a large chunk of data at a time to reduce memory usage. """

    dataloader = get_dataloader(
        input_dataset,
        field=input_field,
        tokenize_fn=model.model.tokenize,
        batch_size=encoding_batch_size,
        remove_stopwords=remove_stopwords,
        max_encoding_length=max_encoding_length,
    )

    # gather all predictions as single non-nested list
    all_predictions = trainer.predict(model, dataloaders=dataloader, return_predictions=True)

    # retrieving uuids and embeddings
    uuids = torch.cat([step_predictions[0] for step_predictions in all_predictions], axis=0)
    embeddings = torch.cat([step_predictions[1] for step_predictions in all_predictions], axis=0)

    # temporarily save to disk to avoid RAM overflows
    cache_filename = f"datasets_augmentation-chunk_{chunk_id}-process_{trainer.global_rank}"
    cache_filepath = os.path.join(tmp_dir, cache_filename)
    torch.save(dict(uuids=uuids, embeddings=embeddings), cache_filepath)

    # assert all processes in distributed saved the embeddings
    if distributed_available():
        torch.distributed.barrier()


def main(args):
    r""" Augment dataset with similar sentences. """
    assert not os.path.exists(args.output_dataset), (
        f"Cannot write to {args.output_dataset} because it is not empty"
    )

    # clean tmp dir from files of this framework
    rank_zero_info("Cleaning tmp directory from files of this framework...")
    cache_file_prefix = 'datasets_augmentation'
    os.makedirs(args.tmp_dir, exist_ok=True)
    clean_folder(args.tmp_dir, cache_file_prefix)

    rank_zero_info("Loading model and moving to device...")
    model = EncodingModel(args.model)

    # all callbacks (custom tqdm progress + model summary)
    callbacks = [RichModelSummary(max_depth=2), TQDMProgressBar(process_position=1)]

    # disallow dp and ddp2 which are going to be deprecated
    assert args.strategy not in ("dp", "ddp2"), (
        "This repo is not designed to work with DataParallel. Use strategy `ddp` or other strategies instead."
    )

    kwargs = dict(callbacks=callbacks, logger=None, default_root_dir=OUTPUT_LOGGING)

    # instantiate PL trainer
    trainer_hyperparameters = get_trainer_args_from_hyperparameters(args)
    trainer = Trainer(**trainer_hyperparameters, **kwargs)

    # data
    original_dataset = prepare_dataset(
        args,
        local_rank=trainer.local_rank,
        global_rank=trainer.global_rank,
        prepare_data_per_node=True,
        shuffle=args.shuffle,
    )

    # start chunked encoding
    rank_zero_info("Encoding input dataset...")
    if args.encoding_chunk_size is None:
        input_datasets = [original_dataset]
    else:
        input_datasets = split_dataset_in_chunks(original_dataset, chunk_size=args.encoding_chunk_size)

    for i, dataset in tqdm(
        enumerate(input_datasets),
        desc="Total progress",
        total=math.ceil(len(original_dataset) / args.encoding_chunk_size),
        disable=trainer.global_rank > 0,
    ):
        encode(
            input_dataset=dataset,
            model=model,
            trainer=trainer,
            input_field=args.input_field,
            encoding_batch_size=args.encoding_batch_size,
            remove_stopwords=args.remove_stopwords,
            max_encoding_length=args.max_encoding_length,
            tmp_dir=args.tmp_dir,
            chunk_id=i,
        )

    if not distributed_available() or trainer.global_rank == 0:
        rank_zero_info("Building final dataset...")

        input_encoding_field = f"{args.input_field}_encoding"
        cache_files = sorted([f for f in os.listdir(args.tmp_dir) if f.startswith(cache_file_prefix)])
        cache_filepaths = [os.path.join(args.tmp_dir, f) for f in cache_files]

        output_dataset = Dataset.from_generator(
            cache_files_reader,
            gen_kwargs=dict(
                cache_files=cache_filepaths, embeddings_name=input_encoding_field, use_multiprocessing=False
            ),
        )

        if args.output_encoding_type is not None:
            rank_zero_info(f"Converting encodings to {args.output_encoding_type}...")
            new_features = copy.deepcopy(output_dataset.features)
            new_features[input_encoding_field] = Sequence(feature=Value(dtype=args.output_encoding_type))
            output_dataset = output_dataset.cast(
                new_features,
                num_proc=cpu_count(),
            )

        rank_zero_info("Sorting dataset to ensure data are in original order...")
        output_dataset = output_dataset.sort('uuid').remove_columns('uuid')

        rank_zero_info("Merging with original dataset...")
        output_dataset = concatenate_datasets([original_dataset, output_dataset], axis=1)

        rank_zero_info("Saving to disk...")
        output_dataset.save_to_disk(args.output_dataset)

        rank_zero_info("Cleaning...")
        clean_folder(args.tmp_dir, cache_file_prefix)

        rank_zero_info(f"Successfully computed embeddings of size {model.get_sentence_embedding_dimension()}!")


if __name__ == "__main__":
    parser = ArgumentParser()

    # input dataset
    parser.add_argument('--input_dataset', type=str, required=True)
    parser.add_argument('--input_field', type=str, required=True, help="Field for which data will be generated")
    parser.add_argument('--input_shard', type=int, required=False, default=None)
    parser.add_argument('--input_limit', type=int, required=False, default=None)
    parser.add_argument('--split_in_sentences', action="store_true")
    parser.add_argument('--shuffle', action="store_true")

    # model to encode sentences
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--remove_stopwords', action="store_true", help="Remove stopwords in encoding.")

    # encoding parameters
    parser.add_argument('--encoding_batch_size', type=int, default=128, required=False)
    parser.add_argument('--encoding_chunk_size', type=int, default=2**23, required=False)
    parser.add_argument('--max_encoding_length', type=int, default=128, required=False)

    # resulting dataset
    parser.add_argument('--output_dataset', type=str, required=True)
    parser.add_argument(
        '--output_encoding_type', type=str, required=False, choices=('float16', 'float32', 'float64'), default=None
    )

    # tmp folders management
    parser.add_argument(
        '--tmp_dir', type=str, required=False, default="/tmp", help="Change if space on main disk is limited"
    )
    parser.add_argument('--reset_cache', action="store_true", help="Clean all previously cached processed datasets")

    add_trainer_args(parser)

    args = ExtendedNamespace.from_namespace(parser.parse_args())
    main(args)
