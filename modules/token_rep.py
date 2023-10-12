from typing import List

import torch
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from torch import nn
from torch.nn.utils.rnn import pad_sequence


# flair.cache_root = '/gpfswork/rech/pds/upa43yu/.cache'


class TokenRepLayer(nn.Module):
    def __init__(self, model_name: str = "bert-base-cased", fine_tune: bool = True, subtoken_pooling: str = "first"):
        super().__init__()

        self.bert_layer = TransformerWordEmbeddings(
            model_name,
            fine_tune=fine_tune,
            subtoken_pooling=subtoken_pooling,
            allow_long_sentences=True
        )

        self.hidden_size = self.bert_layer.embedding_length

    def forward(self, tokens: List[List[str]], lengths: torch.Tensor):
        token_embeddings = self.compute_word_embedding(tokens)
        B = len(lengths)
        max_length = lengths.max()
        mask = (torch.arange(max_length).view(1, -1).repeat(B, 1) < lengths.cpu().unsqueeze(1)).to(
            token_embeddings.device).long()
        return {"embeddings": token_embeddings, "mask": mask}

    def compute_word_embedding(self, tokens):
        sentences = [Sentence(i) for i in tokens]
        self.bert_layer.embed(sentences)
        token_embeddings = pad_sequence([torch.stack([t.embedding for t in k]) for k in sentences], batch_first=True)
        return token_embeddings
