from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def process_function(
    text: List[str],
    rank: int,
    model: str = None,
    batch_size: int = None,
    dtype: torch.dtype = None,
    devices: int = None,
):
    r""" Encode all entries of text in dataset with given model. """
    # rank may be none in single process works
    if rank is None:
        rank = 0
    else:
        rank = rank % devices

    # set device
    device = torch.device(f'cuda:{rank}')

    # instantiate model
    model = SentenceTransformer(model, device=device)

    with torch.autocast(device_type="cuda", dtype=dtype):
        embeddings = model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=False,
            output_value='sentence_embedding',
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

    del model

    return dict(embeddings=embeddings.tolist())
