import logging
from functools import partial
from multiprocessing import cpu_count
from typing import Callable, Dict, List

import torch
from datasets import Dataset, load_from_disk
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
    text: List[str],
    tokenize_fn: Callable = None,
    max_sequence_length: int = None,
    device: torch.device = 'cpu',
) -> Dict:
    r""" Tokenizer batch of sentences. """

    features = tokenize_fn(text)
    features = {k: v.to(device) for k, v in features.items() if isinstance(v, torch.Tensor)}

    # if max_sequence_length is not None:
    #     features = {k: v[:, :max_sequence_length] for k, v in features.items()}
    return features


def get_dataloader(
    text: List[str],
    tokenize_fn: Callable,
    batch_size: int,
    max_sequence_length: int = None,
    num_workers: int = 0,
    device: torch.device = 'cpu',
) -> DataLoader:
    r""" Tokenizer input sentences, create batches and eventually clip to max length. """

    partial_collate_fn = partial(
        collate_fn,
        tokenize_fn=tokenize_fn,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    return DataLoader(
        text,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial_collate_fn,
        pin_memory=True,
        drop_last=False,
    )


def prepare_dataset(
    input_dataset: str,
    input_field: str,
    input_shard: int = None,
    input_limit: int = None,
    reset_cache: bool = False,
    split_in_sentences: bool = False,
) -> Dataset:

    logging.info("Loading input dataset...")
    input_dataset = load_from_disk(input_dataset)

    logging.info("Checking datasets features...")
    assert input_field in input_dataset.features and input_dataset.features[input_field].dtype == 'string'

    logging.info("Sharding and limiting input datasets...")
    input_dataset = limit_and_shard(input_dataset, shard=input_shard, limit=input_limit)

    if reset_cache:
        input_dataset.cleanup_cache_files()

    if split_in_sentences:
        logging.info("Splitting dataset field in single sentences...")
        input_dataset = split_field_in_sentences(input_dataset, field=input_field)
        logging.info(f"New input dataset length is {len(input_dataset)}")

    return input_dataset
