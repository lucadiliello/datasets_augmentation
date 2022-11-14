import logging
import math
import os
from argparse import ArgumentParser
import numpy as np
import faiss
from datasets import load_from_disk
from multiprocess import cpu_count
from tqdm import tqdm

from datasets_augmentation.utilities import flatten_on_field, parse_arg, return_column_data_in_splits, split_dataset


logging.getLogger().setLevel(logging.INFO)


def main(args):
    r""" Augment dataset with similar (negative) candidates. """
    assert not os.path.exists(args.output_dataset), (
        f"Cannot write to {args.output_dataset} because it is not empty"
    )

    if args.flatten_change_fields is not None:
        assert args.flatten is True, "Cannot set `--flatten_change_fields` without `--flatten`"
        args.flatten_change_fields = parse_arg(args.flatten_change_fields)

    assert not args.shard_index or args.addition_batch_size is None, (
        f"Cannot add in batches when using sharded index"
    )

    logging.info("Loading datasets...")
    input_dataset = load_from_disk(args.input_dataset)
    augment_dataset = load_from_disk(args.augment_dataset).shard(100, 0)

    if args.reset_cache:
        input_dataset.cleanup_cache_files()
        augment_dataset.cleanup_cache_files()

    logging.info("Checking datasets features...")
    for field, dataset in zip((args.input_field, args.augment_field), (input_dataset, augment_dataset)):
        encoding_field = f"{field}_encoding"
        assert (
            field in dataset.features
            and dataset.features[field].dtype == 'string'
            and dataset.features[encoding_field].dtype == 'list'
            and dataset.features[encoding_field].feature.dtype.startswith('float32')
        ), (
            f"Field {field} is not a column containing strings or "
            f"field {field}_encoding does not contain encodings as list of floats32. "
            f"Available features are {dataset.features}"
        )

    logging.info(f" - Embeddings are of type {dataset.features[encoding_field].feature.dtype}")

    # names of columns containing encodings
    input_encoding_field = f"{args.input_field}_encoding"
    augment_encoding_field = f"{args.augment_field}_encoding"

    # return directly numpy objects to avoid double conversion to list and back to numpy
    input_dataset.set_format('numpy', columns=[input_encoding_field], output_all_columns=True)
    augment_dataset.set_format('numpy', columns=[augment_encoding_field])

    logging.info("Building index...")
    # search_engine = faiss.IndexScalarQuantizer(args.hidden_size, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
    search_engine = faiss.IndexFlatL2(args.hidden_size)

    if args.devices is not None:
        logging.info("Moving index to GPU(s)...")
        assert args.devices <= faiss.get_num_gpus(), (
            f"Cannot run index on {args.devices} devices, found only {faiss.get_num_gpus()}"
        )
        co = faiss.GpuMultipleClonerOptions()
        co.shard = args.shard_index
        co.useFloat16 = args.fp16
        search_engine = faiss.index_cpu_to_all_gpus(search_engine, co=co, ngpu=args.devices)

    logging.info("Adding data to index...")
    if args.addition_batch_size is not None:
        for adding_batch in tqdm(
            split_dataset(augment_dataset, args.addition_batch_size),
            desc="Adding",
            total=math.ceil(len(augment_dataset) / args.addition_batch_size)
        ):
            data = np.vstack(adding_batch[augment_encoding_field])
            search_engine.add(data)
    else:
        logging.info(" - Collecting embeddings from dataset...")
        data = np.vstack(augment_dataset[augment_encoding_field])
        logging.info(" - Adding embeddings to index...")
        search_engine.add(data)

    logging.info("Finding similar sentences embeddings...")
    queries = np.vstack(input_dataset[input_encoding_field])

    indexes = []
    for part in tqdm(np.array_split(queries, max(len(queries) // args.search_batch_size, 1), axis=0), desc="Quering"):
        indexes.append(search_engine.search(part, k=args.top_k)[1])
    indexes = np.concatenate(indexes, axis=0)

    real_split_sizes = (indexes != -1).sum(-1)
    indexes = indexes.flatten()

    logging.info("Retrieving similar sentences from augmentation dataset...")
    data = list(
        return_column_data_in_splits(
            augment_dataset.select(indexes), column_name=args.augment_field, split_size=real_split_sizes
        )
    )

    # indexes: double array containing indexes of top_k elements for each query
    augmented_field_name = f"{args.input_field}_augmented"
    input_dataset = input_dataset.remove_columns(input_encoding_field).add_column(augmented_field_name, data)

    if args.flatten:
        logging.info("Flattening final dataset...")
        input_dataset = input_dataset.map(
            flatten_on_field,
            num_proc=cpu_count(),
            fn_kwargs=dict(
                field=args.input_field,
                augment_field=augmented_field_name,
                flatten_change_fields=args.flatten_change_fields,
            ),
            batched=True,
            remove_columns=input_dataset.column_names,
        )

    logging.info("Saving results")
    input_dataset.save_to_disk(args.output_dataset)


if __name__ == "__main__":
    parser = ArgumentParser()

    # input dataset
    parser.add_argument('--input_dataset', type=str, required=True)
    parser.add_argument('--input_field', type=str, required=True, help="Field that was previously encoded")

    # dataset to search in for augmentations
    parser.add_argument('--augment_dataset', type=str, required=True)
    parser.add_argument(
        '--augment_field', type=str, required=True, help="Field that was previously encoded"
    )

    parser.add_argument('--devices', type=int, default=None, required=False)
    parser.add_argument('--fp16', action="store_true", help="Use fp16 to store data")

    # encoding parameters
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument(
        '--shard_index', action="store_true", help="Shard index on the available GPUs instead of replicating"
    )
    parser.add_argument('--addition_batch_size', type=int, default=None, required=False)
    parser.add_argument('--search_batch_size', type=int, default=2**14, required=False)
    parser.add_argument('--top_k', type=int, default=10, required=False)
    parser.add_argument('--flatten', action="store_true")
    parser.add_argument(
        '--flatten_change_fields', type=str, nargs='+', default=None, help="Fields to be changed while flattening"
    )

    # resulting dataset path
    parser.add_argument('--output_dataset', type=str, required=True)
    parser.add_argument('--reset_cache', action="store_true", help="Clean all previously cached processed datasets")

    args = parser.parse_args()
    main(args)
