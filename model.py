from typing import List, Dict, Any

import torch
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder

from modules.base import SpanModelBase
from modules.layers import SpanGlobal
from modules.span_rep import SpanRepLayer
from modules.token_rep import TokenRepLayer
from tqdm import tqdm
from modules.evaluator import Evaluator


class NerSpanModel(SpanModelBase):
    def __init__(self, config):
        super().__init__(config.entity_types, config.max_width)

        self.config = config

        # usually a pretrained bidirectional transformer, returns first subtoken representation
        self.token_rep_layer = TokenRepLayer(model_name=config.model_name, fine_tune=config.fine_tune,
                                             subtoken_pooling=config.subtoken_pooling)

        # hierarchical representation of tokens
        self.rnn = LstmSeq2SeqEncoder(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
        )

        # span representation
        self.span_rep_layer = SpanRepLayer(
            span_mode=config.span_mode,
            hidden_size=config.hidden_size,
            max_width=config.max_width,
            dropout=config.dropout,
        )

        # output layer/decoder
        # standard: Span Structured Prediction (Zaratiana et al., UM-IoS 2022)
        # gss: Global Span Selection for Named Entity Recognition (Zaratiana et al., UM-IoS 2022)
        # fsemicrf: Filtered Semi-CRF (Zaratiana et al., EMNLP 2023)
        self.span_global = SpanGlobal(
            hidden_size=config.hidden_size,
            num_classes=len(config.entity_types) + 1,
            sample_rate=config.sample_rate,  # give lower weight to "null" spans
            model_type=config.model_type,  # ['standard', 'gss', 'fsemicrf']
        )

    def get_optimizer(self, config):
        # learning rates
        lr_encoder, lr_other = config.learning_rates
        lr_encoder, lr_other = float(lr_encoder), float(lr_other)

        # weight decay
        weight_decay_encoder, weight_decay_other = config.weight_decay
        weight_decay_encoder, weight_decay_other = float(weight_decay_encoder), float(weight_decay_other)

        # parameters
        params = [
            {"params": self.token_rep_layer.parameters(), "lr": lr_encoder, "weight_decay": weight_decay_encoder},
            {"params": self.rnn.parameters(), "lr": lr_other, "weight_decay": weight_decay_other},
            {"params": self.span_rep_layer.parameters(), "lr": lr_other, "weight_decay": weight_decay_other},
            {"params": self.span_global.parameters(), "lr": lr_other, "weight_decay": weight_decay_other}
        ]
        # return optimizer
        return torch.optim.AdamW(params)

    def compute_span_representation(self, x):
        span_idx = x['span_idx'] * x['span_mask'].unsqueeze(-1)
        out = self.token_rep_layer(x["tokens"], x["seq_length"])
        word_rep = out["embeddings"]
        word_rep = self.rnn(word_rep, out['mask'])
        span_rep = self.span_rep_layer(word_rep, span_idx)
        return span_rep

    def compute_loss(self, x: Dict[str, Any]) -> torch.Tensor:
        span_rep = self.compute_span_representation(x)
        loss = self.span_global.loss(span_rep, span_label=x["span_label"], span_idx=x["span_idx"])
        return loss

    @torch.no_grad()
    def predict(self, x: Dict[str, Any]) -> List[Dict[str, Any]]:
        span_rep = self.compute_span_representation(x)
        output = self.span_global.predict_batch(
            span_rep, span_idx=x["span_idx"],
            mask=x["span_mask"],
            id_to_classes=self.id_to_classes,
            decoding=self.config.decoding
        )
        return output

    def evaluate(self, data_loader):
        self.eval()
        device = next(self.parameters()).device
        all_preds = []
        all_labels = []
        for x in tqdm(data_loader, desc="Evaluating"):
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            preds = self.predict(x)
            all_preds.extend(preds)
            all_labels.extend(x["gold_labels"])

        evaluator = Evaluator(all_labels, all_preds)
        out, f1 = evaluator.evaluate()
        return out, f1