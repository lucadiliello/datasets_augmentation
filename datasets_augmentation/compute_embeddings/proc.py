from multiprocessing import cpu_count
from typing import List

import torch
from datasets_augmentation.compute_embeddings.data import get_dataloader
from datasets_augmentation.compute_embeddings.model import EncodingModel


def process_function(
    text: List[str],
    rank: int,
    model: str = None,
    batch_size: int = None,
    max_sequence_length: int = None,
    dtype: torch.dtype = None,
    devices: int = None,
):
    r""" Encode all entries of text in dataset with given model. """
    # rank may be none in single process works
    if rank is None:
        rank = 0

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    model = EncodingModel(model)

    # set device
    device = torch.device(f'cuda:{rank}')

    # move model to right device and set in eval mode
    print("Instantiating model")
    model = model.eval()
    model = model.to(device)

    # create dataloader
    print("Create dataloader")
    dataloader = get_dataloader(
        text,
        tokenize_fn=model.model.tokenize,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        num_workers=cpu_count() // (devices or 1),  # automatically partition cores for accelerators
        device=device,
    )

    print("Embedding")
    print(model.device)
    res = []
    with torch.inference_mode():
        # with torch.autocast(device_type="cuda", dtype=dtype):
        for batch in dataloader:
            print(batch['input_ids'].device); exit()
            output = model(**batch)
            output = output.cpu().detach().tolist()
            res += output
            print(len(res))
    
    print("Del model")
    del model

    return res
