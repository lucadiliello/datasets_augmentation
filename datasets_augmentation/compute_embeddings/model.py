from typing import Any, Dict, List

import torch
from lightning.pytorch import LightningModule
from sentence_transformers import SentenceTransformer


class EncodingModel(LightningModule):

    def __init__(self, model_path: str):
        super().__init__()
        self.model = SentenceTransformer(model_path)

    def get_sentence_embedding_dimension(self):
        return self.model.get_sentence_embedding_dimension()

    def predict_step(self, batch: Dict, *args: List, **kwargs: Dict) -> Any:
        r""" Does an inference step over the input strings and return embeddings. """
        uuid = batch.pop('uuid')

        out_features = self.model.forward(batch)
        embeddings = torch.nn.functional.normalize(out_features['sentence_embedding'], p=2, dim=1)

        return [uuid, embeddings]
