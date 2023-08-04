import logging
from argparse import Namespace
from functools import partial
from multiprocessing import cpu_count
from typing import Callable, Dict, List

import torch
from datasets import Dataset, load_from_disk
from lightning_fabric.utilities.distributed import _distributed_available as distributed_available
from torch.utils.data import DataLoader

from datasets_augmentation.utilities import remove_stopwords_from_string, split_in_sentences


def limit_and_shard(dataset: Dataset, shard: int = None, limit: int = None) -> Dataset:
    r""" Shard and/or limit input length of dataset. """
    if shard is not None:
        dataset = dataset.shard(shard, 0, contiguous=False)

    if limit is not None:
        dataset = dataset.select(range(limit))

    return dataset


def split_field_in_sentences(dataset: Dataset, field: str) -> Dataset:
    r""" Split content of a field in multiple strings. Dataset length will be thus increased. """
    return dataset.map(
        split_in_sentences,
        num_proc=cpu_count(),
        remove_columns=dataset.column_names,
        batched=True,
        fn_kwargs=dict(field=field),
    )


def collate_fn(
    batch: List[Dict],
    field: str = None,
    tokenize_fn: Callable = None,
    remove_stopwords: bool = False,
    max_encoding_length: int = None,
) -> Dict:
    r""" Tokenizer batch of sentences. """

    sentences_batch = [remove_stopwords_from_string(b[field]) if remove_stopwords else b[field] for b in batch]
    features = tokenize_fn(sentences_batch)

    if max_encoding_length is not None:
        features = {k: v[:, :max_encoding_length] for k, v in features.items()}

    # uuid will be used to rebuild order and retrieve additional original data
    uuid = torch.tensor([b['uuid'] for b in batch])

    return dict(uuid=uuid, **features)


def get_dataloader(
    dataset: Dataset,
    field: str,
    tokenize_fn: Callable,
    batch_size: int,
    remove_stopwords: bool = False,
    max_encoding_length: int = None,
    num_workers: int = 0,
) -> DataLoader:
    r""" Tokenizer input sentences, create batches and eventually clip to max length. """

    dataset = dataset.add_column('uuid', list(range(len(dataset))))

    partial_collate_fn = partial(
        collate_fn,
        field=field,
        tokenize_fn=tokenize_fn,
        remove_stopwords=remove_stopwords,
        max_encoding_length=max_encoding_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial_collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )


def prepare_dataset(
    args: Namespace,
    local_rank: int,
    global_rank: int,
    prepare_data_per_node: bool = False,
    shuffle: bool = False,
) -> Dataset:

    logging.info("Loading input dataset...")
    input_dataset = load_from_disk(args.input_dataset)

    logging.info("Checking datasets features...")
    assert args.input_field in input_dataset.features and input_dataset.features[args.input_field].dtype == 'string'

    logging.info("Sharding and limiting input datasets...")
    input_dataset = limit_and_shard(input_dataset, shard=args.input_shard, limit=args.input_limit)

    # preparing protected environment from concurrency
    rank = local_rank if prepare_data_per_node else global_rank

    if args.reset_cache and rank == 0:
        input_dataset.cleanup_cache_files()

    # entering protected environment from concurrency
    if distributed_available() and rank > 0:
        torch.distributed.barrier()

    if args.split_in_sentences:
        if shuffle:
            logging.info("Shuffling dataset")
            input_dataset = input_dataset.shuffle()

        logging.info("Splitting dataset field in single sentences...")
        input_dataset = split_field_in_sentences(input_dataset, field=args.input_field)
        logging.info(f"New input dataset length is {len(input_dataset)}")

    if distributed_available() and rank == 0:
        torch.distributed.barrier()

    return input_dataset
