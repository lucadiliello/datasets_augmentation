import logging
import os
from argparse import ArgumentParser

import torch

# needed to run gpu computations inside the workers
torch.multiprocessing.set_start_method('spawn')


from datasets_augmentation.compute_embeddings.data import prepare_dataset  # noqa: E402
from datasets_augmentation.compute_embeddings.proc import process_function  # noqa: E402


os.environ['TOKENIZERS_PARALLELISM'] = "true"
logging.getLogger("transformers").setLevel(logging.ERROR)  # too much complains of the tokenizers
logger = logging.getLogger("datasets_augmentation")
logger.setLevel(logging.INFO)


def main(args):
    r""" Augment dataset with similar sentences. """
    assert not os.path.exists(args.output_dataset), (
        f"Cannot write to {args.output_dataset} because it is not empty"
    )

    assert 1 <= args.devices <= torch.cuda.device_count()

    if args.devices == 0:
        args.devices = None

    # preprocess data, split in sentences
    original_dataset = prepare_dataset(
        input_dataset=args.input_dataset,
        input_field=args.input_field,
        input_shard=args.input_shard,
        input_limit=args.input_limit,
        reset_cache=args.reset_cache,
        split_in_sentences=args.split_in_sentences,
    )

    # set dtype
    if args.precision == '16-mixed':
        dtype = torch.float16
    elif args.precision == 'bf16-mixed':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # start chunked encoding
    logger.info("Encoding input dataset...")

    output_dataset = original_dataset.map(
        process_function,
        batched=True,
        with_rank=True,
        batch_size=args.loading_batch_size,
        input_columns=args.input_field,
        num_proc=args.devices,
        fn_kwargs=dict(
            model=args.model,
            batch_size=args.batch_size,
            dtype=dtype,
            devices=args.devices,
        )
    )

    logger.info("Saving to disk...")
    output_dataset.save_to_disk(args.output_dataset)

    logger.info("Successfully computed embeddings!")


if __name__ == "__main__":
    parser = ArgumentParser()

    # input dataset
    parser.add_argument('--input_dataset', type=str, required=True)
    parser.add_argument('--input_field', type=str, required=True, help="Field for which data will be generated")
    parser.add_argument('--input_shard', type=int, required=False, default=None)
    parser.add_argument('--input_limit', type=int, required=False, default=None)
    parser.add_argument('--split_in_sentences', action="store_true")

    # model to encode sentences
    parser.add_argument('--model', type=str, required=True)

    # encoding parameters
    parser.add_argument('--batch_size', type=int, default=2**10, required=False)
    parser.add_argument('--loading_batch_size', type=int, default=2**20, required=False)

    # resulting dataset
    parser.add_argument('--output_dataset', type=str, required=True)

    # tmp folders management
    parser.add_argument('--reset_cache', action="store_true", help="Clean all previously cached processed datasets")

    # add devices and precision
    allowed_prec = ('16-mixed', 'bf16-mixed', '32-true')
    parser.add_argument('--devices', type=int, default=torch.cuda.device_count(), required=False)
    parser.add_argument('--precision', type=str, default=allowed_prec[0], required=False, choices=allowed_prec)

    args = parser.parse_args()
    main(args)
