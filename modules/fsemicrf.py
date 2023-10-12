from torch import nn
from .graph import IntervalGraph, has_overlapping


def make_tuples(idx, label, label_filter, label_score):
    return list(zip(idx.squeeze(), label.squeeze(), label_filter.view(-1), label_score.view(-1)))


def get_gold_segments(span_dict):
    return [(k, l, g) for (k, v, l, g) in span_dict if v.item() > 0]


def overlap_gold(span, gold):
    return any(has_overlapping(tuple(sp), tuple(span)) for sp in gold)


def non_null_segments(span_dict, gold):
    gold = [i[0] for i in gold]
    return [(k, l, g) for (k, v, l, g) in span_dict if v.item() == 0 and l.item() > 0 and overlap_gold(k, gold)]


def get_spans_scores(idx, label, label_filter, label_score):
    # create tuple and apply filtering
    span_dict = make_tuples(idx, label, label_filter, label_score)
    gold_segments = get_gold_segments(span_dict)
    valid_segments = non_null_segments(span_dict, gold_segments)

    filtered_segments, filtered_scores, filtered_labels = [], [], []

    for i in gold_segments + valid_segments:
        filtered_segments.append(tuple(i[0].tolist()))
        filtered_labels.append(i[1].item())
        filtered_scores.append(i[2])

    # gold segments (for nll computation)
    only_gold = [tuple(i[0].tolist()) for i in gold_segments]
    return only_gold, filtered_segments, filtered_scores, filtered_labels


class FilteredSemiCRFLoss(nn.Module):
    def forward(self, all_segment_idx, all_segment_label, all_label_filter, all_scores, transition_score):
        batch_size = len(all_label_filter)
        total_loss = 0

        for b in range(batch_size):
            gold_segments, filtered_segments, filtered_scores, filtered_labels = get_spans_scores(
                idx=all_segment_idx[b],
                label=all_segment_label[b],
                label_filter=all_label_filter[b],
                label_score=all_scores[b]
            )
            if len(gold_segments) < 1 or len(filtered_segments) == len(gold_segments):
                continue

            int_graph = IntervalGraph(spans=filtered_segments, scores=filtered_scores, labels=filtered_labels,
                                      transition_matrix=transition_score, is_sorted=False)

            # negative likelihood (nll) of the gold path
            loss = int_graph.compute_path_nll(gold_segments)
            total_loss += loss

        return total_loss
