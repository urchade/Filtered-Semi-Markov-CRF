from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class SpanModelBase(nn.Module):
    def __init__(self, entity_types: List[str], max_width: int):
        super().__init__()

        # Tags: BIO
        self.classes_to_id = {c: i for i, c in enumerate(entity_types, start=1)}
        self.classes_to_id['O'] = 0
        self.id_to_classes = {k: v for v, k in self.classes_to_id.items()}
        self.max_width = max_width
        self.num_classes = len(self.classes_to_id)

    def preprocess(self, tokens: List[str], spans: List[Tuple[int, int, str]]):
        """
        Preprocess spans for model input
        :param tokens: list of tokens (str)
        :param spans: list of spans (start, end, label)
        :return: dict with keys: tokens, span_idx, span_label, seq_length
        """

        spans_idx = [(i, i + j) for i in range(len(tokens)) for j in range(self.max_width)]  # get all spans

        # dict that maps span to label
        dict_lab = defaultdict(int)
        for span in spans:
            # span: (start, end, label)
            dict_lab[(span[0], span[1])] = self.classes_to_id[span[2]]

        # Get labels for spans
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])  # (num_spans, )
        spans_idx = torch.LongTensor(spans_idx)  # (num_spans, 2)

        # Mask for valid spans
        valid_span_mask = spans_idx[:, 1] > len(tokens) - 1  # (num_spans, )

        # Mask invalid positions
        span_label = span_label.masked_fill(valid_span_mask, -1)  # (num_spans, )

        return {
            'tokens': tokens,
            'span_idx': spans_idx,
            'span_label': span_label,
            'seq_length': len(tokens),
            'gold_labels': spans
        }

    def collate_fn(self, batch_list: List[Dict]):
        batch = [self.preprocess(tokens, labels) for tokens, labels in batch_list]
        span_idx = pad_sequence([b['span_idx'] for b in batch], batch_first=True, padding_value=0)
        span_label = pad_sequence([el['span_label'] for el in batch], batch_first=True, padding_value=-1)
        span_mask = span_label != -1

        return {
            'seq_length': torch.LongTensor([el['seq_length'] for el in batch]),
            'span_idx': span_idx,
            'tokens': [el['tokens'] for el in batch],
            'gold_labels': [el['gold_labels'] for el in batch],
            'span_mask': span_mask,
            'span_label': span_label,
        }

    def create_dataloader(self, data, **kwargs):
        return DataLoader([i.values() for i in data], collate_fn=self.collate_fn, **kwargs)
