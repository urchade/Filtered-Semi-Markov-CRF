from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from .fsemicrf import FilteredSemiCRFLoss
from .graph import IntervalGraph
from .losses import down_weight_loss


class SpanGlobal(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, sample_rate: float = 0.5, model_type: str = 'gss',
                 transition: bool = False):
        super().__init__()
        # mode = 'gss' (global span selection) or 'fsemicrf' (filtered semi-crf)
        self.sample_rate = sample_rate
        self.model_type = model_type
        self.num_classes = num_classes

        # Span Structured Prediction (Zaratiana et al., UM-IoS 2022)
        # Local classifier with local score
        if model_type == 'standard':
            self.scorer = nn.Linear(hidden_size, num_classes)
            transition = False

        # Global Span Selection for Named Entity Recognition (Zaratiana et al., UM-IoS 2022)
        # Local classifier + global score (same score for all spans)
        if model_type == 'gss':
            self.scorer = nn.Linear(hidden_size, num_classes + 1)

        # Filtered Semi-CRF (Zaratiana et al., EMNLP 2023)
        # Local classifier + global score (score is label-dependent)
        elif model_type == 'fsemicrf':
            self.scorer = nn.Linear(hidden_size, num_classes * 2)

        if transition:  # transition score for filtered semi-crf
            var = torch.zeros((num_classes + 2, num_classes + 2))  # +2 for start and end transitions
            self.transition_score = nn.Parameter(var)
        else:
            self.transition_score = None

    def compute_scores(self, span_rep: torch.Tensor):
        scores = self.scorer(span_rep)
        B, L, K, C = scores.shape
        scores = scores.view(B, L*K, C)

        local_scores = scores[:, :, :self.num_classes]
        local_labels = local_scores.max(-1).indices.unsqueeze(-1)
        global_scores = None

        if self.model_type == 'standard':
            # use local score as global score
            global_scores = torch.gather(local_scores, -1, local_labels).squeeze(-1)
        elif self.model_type == 'gss':
            # same score for all spans
            global_scores = scores[:, :, -1]
        elif self.model_type == 'fsemicrf':
            # score is label-dependent
            global_scores = torch.gather(scores[:, :, self.num_classes:], -1, local_labels).squeeze(-1)

        return local_scores, global_scores, local_labels

    def loss(self, span_rep: torch.Tensor, span_label: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        # compute scores
        local_scores, global_scores, local_labels = self.compute_scores(span_rep)

        # filtering loss
        loss_local = down_weight_loss(local_scores, span_label, sample_rate=self.sample_rate)

        # Filtered Semi-CRF (Zaratiana et al., EMNLP 2023)
        if self.model_type == 'fsemicrf' or self.model_type == 'gss':
            # loss function for filtered semi-crf and gss
            fsemicrf_loss_func = FilteredSemiCRFLoss()
            # compute loss
            loss_global = fsemicrf_loss_func(
                all_segment_idx=span_idx,
                all_segment_label=span_label,
                all_label_filter=local_labels,
                all_scores=global_scores,
                transition_score=self.transition_score,
            )
        else:  # standard model does not use global score
            loss_global = 0

        return loss_local + loss_global

    @torch.no_grad()
    def predict_batch(self, span_rep: torch.Tensor, span_idx: torch.Tensor, mask: torch.Tensor,
                      id_to_classes: Dict[int, str], decoding: str = 'best') -> List[Dict[str, Any]]:
        """
        Make predictions for spans.

        Parameters:
            span_rep (torch.Tensor): Tensor containing span representations.
            span_idx (torch.Tensor): Tensor containing span indices.
            mask (torch.Tensor): Mask tensor to filter invalid spans.
            id_to_classes (Dict[int, str]): Dictionary mapping label indices to label strings.
            decoding (str): Decoding method to use.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing span predictions, labels, and scores.
        """

        _, scores, labels = self.compute_scores(span_rep)

        # get all predictions
        # contains a list of dictionaries, each dictionary contains the span prediction, label, and score
        # keys for each element: 'spans', 'labels', 'scores'
        all_predictions = []

        for i in range(labels.size(0)):
            label_slice = labels[i].view(-1)

            # Skip if no non-O spans are present
            if torch.where(label_slice > 0)[0].nelement() == 0:
                all_predictions.append([])
                continue

            # Mask out invalid and non-interesting spans
            valid_span_mask = mask[i] * (label_slice > 0)
            valid_spans = span_idx[i]
            pred_labels = torch.masked_select(label_slice, valid_span_mask).tolist()

            # Filter and list valid spans
            span_list = torch.masked_select(valid_spans, valid_span_mask.unsqueeze(-1)).view(-1, 2).tolist()
            span_list = [tuple(item) for item in span_list]

            # Filter and list valid scores
            span_scores = torch.masked_select(scores[i].view(-1), valid_span_mask)

            # combine spans, labels, and scores into a dictionary and decode
            prediction = {'spans': span_list, 'labels': pred_labels, 'scores': span_scores}
            decoded_prediction = self.decode_prediction(prediction, id_to_classes, decoding=decoding)

            # Append to list the spans, labels, and scores
            all_predictions.append(decoded_prediction)

        return all_predictions

    def decode_prediction(self, prediction, id_to_classes, decoding="best"):

        # get spans, labels, and scores
        spans, labels, scores = prediction.values()

        # create dict: span -> label
        span_lab_dict = {span: id_to_classes[label] for span, label in zip(spans, labels)}

        if len(spans) == 0:
            return []

        # make interval graph
        interval_graph = IntervalGraph(spans, scores, labels=labels, transition_matrix=self.transition_score)

        # if no decoding, return spans as is
        output_spans = spans

        # decode
        if decoding in ["best", "gss", "fsemicrf", "global"]:
            # get set of spans with highest score
            output_spans = interval_graph.best_path()
        elif decoding == "greedy":
            # get best span iteratively
            output_spans = interval_graph.greedy_search()
        elif decoding == "global_mean":
            # get set of spans with highest average score
            output_spans = interval_graph.exhaustive_search(scoring_func=np.mean)

        # get labels
        output_labels = [span_lab_dict[span] for span in output_spans]

        # make out put in the format (start, end, label)
        outputs = [(span[0], span[1], label) for span, label in zip(output_spans, output_labels)]

        return outputs
