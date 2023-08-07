import logging
from functools import partial
from multiprocessing import cpu_count
from typing import Callable, Dict, List

import torch
from datasets import Dataset, load_from_disk
from lightning_fabric.utilities.distributed import _distributed_available as distributed_available
from torch.utils.data import DataLoader

from datasets_augmentation.utilities import split_in_sentences


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
    max_sequence_length: int = None,
) -> Dict:
    r""" Tokenizer batch of sentences. """

    sentences_batch = [b[field] for b in batch]
    features = tokenize_fn(sentences_batch)

    if max_sequence_length is not None:
        features = {k: v[:, :max_sequence_length] for k, v in features.items()}

    # uuid will be used to rebuild order and retrieve additional original data
    # uuid = torch.tensor([b['uuid'] for b in batch])
    return features


def get_dataloader(
    dataset: Dataset,
    field: str,
    tokenize_fn: Callable,
    batch_size: int,
    max_sequence_length: int = None,
    num_workers: int = 0,
) -> DataLoader:
    r""" Tokenizer input sentences, create batches and eventually clip to max length. """

    # dataset = dataset.add_column('uuid', list(range(len(dataset))))

    partial_collate_fn = partial(
        collate_fn,
        field=field,
        tokenize_fn=tokenize_fn,
        max_sequence_length=max_sequence_length,
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
    input_dataset: str,
    input_field: str,
    input_shard: int = None,
    input_limit: int = None,
    reset_cache: bool = False,
    split_in_sentences: bool = False,
    global_rank: int = None,
    shuffle: bool = False,
) -> Dataset:

    logging.info("Loading input dataset...")
    input_dataset = load_from_disk(input_dataset)

    logging.info("Checking datasets features...")
    assert input_field in input_dataset.features and input_dataset.features[input_field].dtype == 'string'

    logging.info("Sharding and limiting input datasets...")
    input_dataset = limit_and_shard(input_dataset, shard=input_shard, limit=input_limit)

    # preparing protected environment for concurrency
    if reset_cache and global_rank == 0:
        input_dataset.cleanup_cache_files()

    # entering protected environment for concurrency
    if distributed_available() and global_rank > 0:
        torch.distributed.barrier()

    if shuffle:
        logging.info("Shuffling dataset")
        input_dataset = input_dataset.shuffle()

    if split_in_sentences:
        logging.info("Splitting dataset field in single sentences...")
        input_dataset = split_field_in_sentences(input_dataset, field=input_field)
        logging.info(f"New input dataset length is {len(input_dataset)}")

    if distributed_available() and global_rank == 0:
        torch.distributed.barrier()

    return input_dataset
