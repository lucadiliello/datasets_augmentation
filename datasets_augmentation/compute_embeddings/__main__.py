import hashlib
import logging
import os
import shutil
from argparse import ArgumentParser, Namespace
from multiprocessing import cpu_count
from typing import Dict, Generator

import datasets
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from lightning.fabric import Fabric
from lightning_fabric.utilities.distributed import _distributed_available as distributed_available
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_augmentation.compute_embeddings.data import get_dataloader, prepare_dataset
from datasets_augmentation.compute_embeddings.model import EncodingModel
from datasets_augmentation.utilities import logging_info, rank_zero_info


# configure libraries
os.environ['TOKENIZERS_PARALLELISM'] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)  # too much complains of the tokenizers
torch.set_float32_matmul_precision('medium')


def get_fabric_args_from_hyperparameters(hyperparameters: Namespace) -> Dict:
    r""" Just extract generation hyperparameters from namespace. """
    return dict(
        accelerator=hyperparameters.accelerator,
        strategy=hyperparameters.strategy,
        devices=hyperparameters.devices,
        num_nodes=hyperparameters.num_nodes,
        precision=hyperparameters.precision,
    )


def encode(model: EncodingModel, dataloader: DataLoader, rank: int) -> Generator:
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Encoding({rank})", total=len(dataloader), position=rank):
            embeddings = model.predict_step(batch)
            embeddings = embeddings.cpu().detach().tolist()
            for embedding in embeddings:
                yield dict(embeddings=embedding)


def main(args):
    r""" Augment dataset with similar sentences. """
    assert not os.path.exists(args.output_dataset), (
        f"Cannot write to {args.output_dataset} because it is not empty"
    )

    assert args.num_nodes == 1, "This software works only on a single node with multiple GPUs"

    # clean tmp dir from files of this framework
    rank_zero_info("Creating tmp directories from files of this framework...")
    name = hashlib.sha256((args.model + args.input_dataset).encode()).hexdigest()[:32]
    datasets_cache_folder = os.path.join(args.tmp_dir, name, 'datasets')

    rank_zero_info(f"Experiment code: {name}")

    shutil.rmtree(datasets_cache_folder, ignore_errors=True)
    os.makedirs(datasets_cache_folder, exist_ok=True)

    rank_zero_info("Loading model...")
    model = EncodingModel(args.model)
    embedding_size = model.get_sentence_embedding_dimension()

    rank_zero_info(f"Model produces embeddings with size {embedding_size}")

    # instantiate PL trainer
    fabric_hyperparameters = get_fabric_args_from_hyperparameters(args)
    fabric = Fabric(**fabric_hyperparameters)
    fabric.launch()

    # data
    rank_zero_info("Preparing dataset")
    original_dataset = prepare_dataset(
        input_dataset=args.input_dataset,
        input_field=args.input_field,
        input_shard=args.input_shard,
        input_limit=args.input_limit,
        reset_cache=args.reset_cache,
        split=args.split,
        global_rank=fabric.global_rank,
        paragraph_split_character=args.paragraph_split_character,
    )

    if args.stats and fabric.global_rank == 0:
        rank_zero_info("Computing statistics of prepared dataset...")
        stats = original_dataset.shard(100, 0).map(
            lambda a: dict(length=len(a)), input_columns=args.input_field, num_proc=cpu_count()
        )
        lengths = stats['length']
        rank_zero_info(f"Average length (chars): {np.mean(lengths)}")
        rank_zero_info(f"Std.dev. length (chars): {np.std(lengths)}")
        rank_zero_info(f"Min length (chars): {np.min(lengths)}")
        rank_zero_info(f"Max length (chars): {np.max(lengths)}")
        del lengths, stats

    # disable logging since we have already our fabric logging
    datasets.logging.disable_progress_bar()

    # start chunked encoding
    rank_zero_info("Dividing dataset across processes")
    input_dataset = original_dataset.shard(args.devices, fabric.global_rank, contiguous=True)

    rank_zero_info("Creating dataloader")
    dataloader = get_dataloader(
        input_dataset,
        field=args.input_field,
        tokenize_fn=model.model.tokenize,
        batch_size=args.batch_size,
        max_sequence_length=args.max_sequence_length,
        num_workers=cpu_count() // args.devices,
    )

    # setup model and dataloader
    model = fabric.setup_module(model)
    dataloader = fabric.setup_dataloaders(dataloader, move_to_device=True, use_distributed_sampler=False)

    # embed!
    res = Dataset.from_generator(encode, gen_kwargs=dict(model=model, dataloader=dataloader, rank=fabric.global_rank))

    # unload model and free gpus
    model.to(device=torch.device('cpu'))  # needed because fabric may create new references to the model
    del model

    cache_filename = f"slice_{fabric.global_rank}_of_{fabric.world_size}"
    cache_filepath = os.path.join(datasets_cache_folder, cache_filename)

    logging_info(f"Saving embeddings ({fabric.global_rank}/{fabric.world_size})...", rank=fabric.global_rank)
    res.save_to_disk(cache_filepath)

    # wait for all processes to have saved the embeddings
    rank_zero_info("Waiting for all processes to finish...")
    if distributed_available():
        logging_info("Complete!")
        fabric.barrier()

    # final dataset reconstruction
    if not distributed_available() or fabric.global_rank == 0:
        rank_zero_info("Building final dataset...")

        cache_files = [f"slice_{i}_of_{fabric.world_size}" for i in range(fabric.world_size)]
        cache_filepaths = [os.path.join(datasets_cache_folder, f) for f in cache_files]

        rank_zero_info("Loading computed embeddings...")
        output_embeddings = concatenate_datasets([load_from_disk(f) for f in cache_filepaths], axis=0)

        rank_zero_info("Merging embeddings with original dataset...")
        output_dataset = concatenate_datasets([original_dataset, output_embeddings], axis=1)

        rank_zero_info(f"Saving to disk at '{args.output_dataset}'")
        output_dataset.save_to_disk(args.output_dataset)

        rank_zero_info("Cleaning...")
        for path in cache_filepaths:
            shutil.rmtree(path)

        rank_zero_info(f"Successfully computed embeddings of size {embedding_size}!")


if __name__ == "__main__":
    parser = ArgumentParser()

    # fabric args
    allowed_prec = ('16-mixed', 'bf16-mixed', '32-true')
    parser.add_argument('--accelerator', type=str, default="auto", required=False)
    parser.add_argument('--strategy', type=str, default="auto", required=False)
    parser.add_argument('--devices', type=int, default="auto", required=False)
    parser.add_argument('--num_nodes', type=int, default=1, required=False)
    parser.add_argument('--precision', type=str, default=allowed_prec[0], required=False, choices=allowed_prec)

    # input dataset
    parser.add_argument('--input_dataset', type=str, required=True)
    parser.add_argument('--input_field', type=str, required=True, help="Field for which data will be generated")
    parser.add_argument('--input_shard', type=int, required=False, default=None)
    parser.add_argument('--input_limit', type=int, required=False, default=None)
    parser.add_argument('--split', type=str, required=False, default=None, choices=('sentences', 'paragraphs'))
    parser.add_argument('--paragraph_split_character', type=str, required=False, default='\n\n')
    parser.add_argument('--stats', action="store_true")

    # model to encode sentences
    parser.add_argument('--model', type=str, required=True)

    # encoding parameters
    parser.add_argument('--batch_size', type=int, default=1024, required=False)
    parser.add_argument('--max_sequence_length', type=int, default=128, required=False)

    # resulting dataset
    parser.add_argument('--output_dataset', type=str, required=True)

    # tmp folders management
    parser.add_argument(
        '--tmp_dir',
        type=str,
        required=False,
        default="/science/lucadiliello/.cache/datasets_augmentation",
        help="Change if space on disk is limited",
    )
    parser.add_argument('--reset_cache', action="store_true", help="Clean all previously cached processed datasets")

    args = parser.parse_args()
    main(args)
