from functools import partial
from multiprocessing import cpu_count
from typing import Callable, Dict, List

import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

from datasets_augmentation.utilities import (
    distributed_available,
    rank_zero_info,
    split_in_paragraphs,
    split_in_sentences,
)


def limit_and_shard(dataset: Dataset, shard: int = None, limit: int = None) -> Dataset:
    r""" Shard and/or limit input length of dataset. """
    if shard is not None:
        dataset = dataset.shard(shard, 0, contiguous=False)

    if limit is not None:
        dataset = dataset.select(range(limit))

    return dataset


def split_documents_in_sentences(dataset: Dataset, field: str, **kwargs) -> Dataset:
    r""" Split content of a field in multiple sentences. Dataset length will be thus increased. """
    return dataset.map(
        split_in_sentences,
        num_proc=cpu_count(),
        remove_columns=dataset.column_names,
        batched=True,
        fn_kwargs=dict(field=field, **kwargs),
    )


def split_documents_in_paragraphs(dataset: Dataset, field: str, **kwargs) -> Dataset:
    r""" Split content of a field in multiple paragraphs. Dataset length will be thus increased. """
    return dataset.map(
        split_in_paragraphs,
        num_proc=cpu_count(),
        remove_columns=dataset.column_names,
        batched=True,
        fn_kwargs=dict(field=field, **kwargs),
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
    split: str = None,
    global_rank: int = None,
    **kwargs: Dict,
) -> Dataset:
    r""" Load, check and parse dataset. """

    rank_zero_info("Loading input dataset...")
    input_dataset = load_from_disk(input_dataset)

    rank_zero_info("Checking datasets features...")
    assert input_field in input_dataset.features and input_dataset.features[input_field].dtype == 'string'

    rank_zero_info("Sharding and limiting input datasets...")
    input_dataset = limit_and_shard(input_dataset, shard=input_shard, limit=input_limit)

    # preparing protected environment for concurrency
    if reset_cache and global_rank == 0:
        input_dataset.cleanup_cache_files()

    # entering protected environment for concurrency
    if distributed_available() and global_rank > 0:
        torch.distributed.barrier()

    if split is not None:
        if split == 'sentences':
            rank_zero_info("Splitting dataset documents into sentences...")
            input_dataset = split_documents_in_sentences(input_dataset, field=input_field, **kwargs)
        else:
            rank_zero_info("Splitting dataset documents into paragraphs...")
            input_dataset = split_documents_in_paragraphs(input_dataset, field=input_field, **kwargs)

        rank_zero_info(f"New input dataset length is {len(input_dataset)}")

    if distributed_available() and global_rank == 0:
        torch.distributed.barrier()

    return input_dataset
