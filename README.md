# Datasets Augmentation Toolkit
Increment datasets size by retrieving similar sentences from large corpora.

This library is based on:
- [`sentence-transformers`](https://www.sbert.net)
- [`faiss`](https://faiss.ai)
- [`transformers`](https://huggingface.co/docs/transformers/index) and [`datasets`](https://huggingface.co/docs/datasets/index)
- [`pytorch-lightning`](https://pytorch-lightning.readthedocs.io/en/latest/)
- [`transformers-lightning`](https://github.com/iKernels/transformers-lightning)


The process of retrieving similar sentences is divided into 2 tasks:
- Encoding of the columns containing the sentences to compare
- Comparison and retrieval of the similar sentences


## Encoding

The encoding takes a Huggingface `datasets.Dataset` in input and encodes one column to a vector of floats.

To launch the encoding, run the following command:

```bash
python -m datasets_augmentation.compute_embeddings ... 
```

with the following parameters:
- `--input_dataset <path>`: path to the dataset that will be encoded;
- `--input_field <name_of_the_column>`: the column that will be encoded (must contain strings);
- `--input_shard <n>`: reduces dataset size `n` times (useful for debugging);
- `--input_limit <n>`: limits dataset length to `n` (useful for debugging);
- `--split_in_sentences`: if the dataset contains paragraphs or documents, it may be useful to first split every row into single sentences (thus increasing the dataset length);

- `--model <name_or_path>`: name or path of the `sentence-transformers` model that will be used to encode data;
- `--batch_size <batch_size>`: batch size (per device) for the encoding;
- `--max_sequence_length <length>`: will clip every encoded sentence to this number of tokens;

- `--output_dataset <path>`: path to save the new dataset containing the encodings;

A non-exhaustive list of additional parameters derived from `pytorch-lightning`:
- `--devices <n>`: number of CPUs or GPUs to use;
- `--accelerator <cpu|gpu|tpu|mps|...>`: the device to use for encoding;
- `--strategy <name_of_strategy>`: I strongly suggest `ddp` or `deepspeed_stage_2` if the number of devices is greater than 1, blank otherwise;
- `--precision <32|16>`: use normal `fp32` encoding or speed up training with `fp16`;

The resulting dataset will have an additional column called `{input_field}_encoded` containing the encodings.


### Example

Download the MNLI dataset from the HuggingFace hub:

```python
from datasets import load_dataset
load_dataset('lucadiliello/mnli', split='train').save_to_disk('/path/to/save/dataset')
```

Run encoding of the column `hypothesis` with model `sentence-transformers/nli-roberta-base-v2`, removing stopwords and on a single GPU:
```bash
python -m datasets_augmentation.compute_embeddings \
    --input_dataset /path/to/save/dataset \
    --output_dataset /path/to/save/dataset_encoded \
    --model sentence-transformers/nli-roberta-base-v2 \
    --input_field hypothesis \
    --batch_size 256 \
    --max_sequence_length 128 \
    --devices 1 \
    --accelerator gpu \
```

Now explore the embeddings loading the dataset:

```python
from datasets import load_from_disk
d = load_from_disk('/path/to/save/dataset_encoded')

print(d[0])
print(d[1])
...
```

You should repeat this operation both for the dataset you want to augment and for the dataset from which you will retrieve new sentences.


## Retrival of similar sentences

Run the addition of similar sentences with:

```bash
python -m datasets_augmentation.retrieve_sentences ...
```

with the following parameters:
- `--input_dataset <path>`: path to the dataset that will receive new sentences;
-- `--input_field <name_of_the_column>`: the column that will be used for similarity. A column named `{name_of_the_column}_encoding` should also be present in the `input_dataset` that should have been generated in the previous section;

- `--augment_dataset <path>`: path to the dataset that will be used to retrieve new sentences;
-- `--augment_field <name_of_the_column>`: the column that will be used for similarity. A column named `{name_of_the_column}_encoding` should also be present in the `augment_dataset` that should have been generated in the previous section;

- `--devices <num_devices>`: number of GPUs that should be used for similarity computation with FAISS;
- `--hidden_size <d>`: hidden size of the encodings. It has been printed by the previous method at the end of the encoding;
- `--shard_index`: use this parameter to shard the index over the GPUs instead of replicating it. This will reduce GPU memory usage but also reduce speed;
- `--search_batch_size <batch_size>`: search batch size inside the index;
- `--top_k <k>`: the number of similar sentences that should be retrieved for each 

- `--output_dataset <path>`: path to save the new dataset with additional sentences;

Additional parameters:
- `--flatten`: whether output dataset should be flattened, which means that a new example will be created for each new sentence. Other values will be copied from the pilot example;
- `--flatten_change_fields`: if you want to change some field (for example `label`) only for the new examples when flattening, use this parameter. Use the format `column_name:type:value` to instruct the framework about how to change samples. For example, `label:int:-1` will change the label field to integer -1 for all new samples;


### Example

Retrieve 5 similar sentences for each `hypothesis` inside the `/path/to/save/dataset_encoded` dataset.

```bash
python -m datasets_augmentation.retrieve_sentences \
    --input_dataset /path/to/save/dataset_encoded \
    --input_field hypothesis \
    --augment_dataset /path/to/save/dataset_encoded \
    --augment_field hypothesis \
    --devices 1 \
    --hidden_size 768 \
    --top_k 5 \
    --output_dataset /path/to/save/dataset_augmented \
    --flatten \
    --flatten_change_fields label:int:-100 \
```


## Next steps
- Allow working with `datasets.DatasetDict` instead of only `datasets.Dataset`;


## Troubleshooting

If the `datasets` sizes do not match, maybe an old cached version has been loaded. Try `--reset_cache` to solve or clean HF cache folder `rm -r ~/.cache/huggingface/datasets`;
