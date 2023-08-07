import os
import shutil
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Generator, Iterable, List, Tuple

import nltk
import torch
from blingfire import text_to_sentences
from datasets import Dataset
from lightning.pytorch.callbacks import TQDMProgressBar as PLTQDMProgressBar
from lightning_fabric.utilities.distributed import _distributed_available as distributed_available  # noqa: F401
from lightning_utilities.core.rank_zero import _info, rank_prefixed_message, rank_zero_info  # noqa: F401


def logging_info(message: str, rank: int = None):
    if rank is not None:
        message = rank_prefixed_message(message, rank)
    _info(message)


nltk.download('stopwords', quiet=True)


def dict2list(data: Dict[Any, List]) -> List[Dict]:
    r""" Convert a dict or lists to a list of dicts. """
    values = list(data.values())
    assert all(isinstance(v, Iterable) for v in values)
    assert all(len(v) == len(values[0]) for v in values)

    if not data or all(len(v) == 0 for v in values):
        return []

    keys = data.keys()
    res = [
        {a: b for a, b in zip(keys, values)}
        for values in zip(*[data[key] for key in keys])
    ]
    return res


def list2dict(data: List[Dict]) -> Dict[Any, List]:
    r""" Convert a list of dicts to a dict of lists. """
    if not data:
        return {}

    assert all(isinstance(d, dict) for d in data)
    keys = data[0].keys()
    assert all(d.keys() == keys for d in data)

    res = {k: [d[k] for d in data] for k in keys}
    return res


def split_in_sentences(sample: Dict[str, List], field: str = None, min_sentence_length: int = 10) -> Dict:
    r""" Split text in multiple sentences. """
    res = sum([text_to_sentences(line).split("\n") for line in sample[field] if len(line) >= min_sentence_length], [])
    return {field: res}


def remove_stopwords_from_string(sentence: str) -> str:
    return " ".join(word for word in sentence.split(" ") if word not in nltk.corpus.stopwords.words('english'))


def split_dataset_in_chunks(dataset: Dataset, chunk_size: int = None):
    r""" Split dataset in smaller chunks using select. """
    if chunk_size is None:
        yield dataset
    else:
        assert chunk_size >= 1
        dataset_length = len(dataset)
        start = 0
        while True:
            if start >= dataset_length:
                break
            yield dataset.select(range(start, min(dataset_length, start + chunk_size)))
            start += chunk_size


class TQDMProgressBar(PLTQDMProgressBar):

    @property
    def predict_description(self) -> str:
        return "Chunk progress"


def _flatten_on_field(example: Dict, field: str, augment_field: str, flatten_change_fields: Dict) -> Iterable[Dict]:
    r""" Flat example along field. """

    original_field_data = example.pop(field)
    additional_field_data = example.pop(augment_field)

    yield {field: original_field_data, **example}
    for f in additional_field_data:
        res = {field: f, **example}
        if flatten_change_fields is not None:
            for k, v in flatten_change_fields.items():
                res[k] = v
        yield res


def flatten_on_field(
    examples: Dict[str, List],
    field: str = None,
    augment_field: str = None,
    flatten_change_fields: Dict[str, Any] = None,
) -> Dict:
    r""" Flat examples along field. """
    examples = dict2list(examples)
    examples = [x for example in examples for x in _flatten_on_field(
        example, field=field, augment_field=augment_field, flatten_change_fields=flatten_change_fields
    )]
    return list2dict(examples)


def return_column_data_in_splits(dataset: Dataset, column_name: str, split_size: List[int]) -> Iterable:
    r""" Return data into column as list of size `split_size`. """
    dataset_iterable = iter(dataset[column_name])
    for split in split_size:
        yield [next(dataset_iterable) for _ in range(split)]


def parse_arg(arguments: List[str]) -> Dict[str, Any]:
    r""" Parse argument triples of string:type:value. Value will be converted to type val"""
    res = dict()
    for part in arguments:
        column_name, typ, value = part.split(":")
        res[column_name] = eval(typ)(value)
    return res


def clean_folder(folder: str, prefix: str = None):
    r""" Remove all files in folder. If prefix is not None, remove only files beginning with prefix. """
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if prefix is None or filename.startswith(prefix):
                shutil.rmtree(os.path.join(folder, filename))


def cache_file_reader(inputs: Tuple[str]) -> List[Dict]:
    r""" Read single file in the cache folder and yield single examples. """
    filename, embeddings_name = inputs

    data = torch.load(filename)
    return [
        {'uuid': uuid.item(), embeddings_name: embeddings.tolist()}
        for uuid, embeddings in zip(data['uuids'], data['embeddings'])
    ]


def cache_files_reader(
    cache_files: List[str] = None,
    embeddings_name: str = None,
    use_multiprocessing: bool = False,
) -> Generator[Dict, None, None]:
    r""" Multiprocessing read of files in the cache folder and yield single examples. """
    if use_multiprocessing:
        with Pool(processes=min(cpu_count() // 4, len(cache_files))) as pool:
            for results in pool.map(cache_file_reader, ((filename, embeddings_name) for filename in cache_files)):
                yield from results
    else:
        for filename in cache_files:
            yield from cache_file_reader((filename, embeddings_name))


def split_dataset(dataset: Dataset, split_size: int) -> Iterable[Dataset]:
    r""" Split a dataset on the rows and return iterable of smaller datasets. """
    start = 0
    while start < len(dataset):
        yield dataset.select(range(start, min(len(dataset), start + split_size)))
        start += split_size
