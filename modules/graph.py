from collections import defaultdict

import networkx as nx
import numpy as np
import torch


def has_overlapping(idx1, idx2):
    overlapping = True
    if (idx1[0] > idx2[1] or idx2[0] > idx1[1]):
        overlapping = False
    return overlapping

def create_graph(spans, scores, labels=None, transition_matrix=None, start_trans_index=-2, end_trans_index=-1):
    if labels is None or transition_matrix is None:
        labels = defaultdict(int)
        transition_matrix = defaultdict(int)

    # create interval digraph
    edges = []

    transition_edge = {}

    for i, s_i in enumerate(spans, start=1):
        for j, s_j in enumerate(spans, start=1):
            r_i = s_i[1]
            l_j = s_j[0]
            if r_i < l_j:
                include = True
                for k, s_k in enumerate(spans, start=1):
                    l_k, r_k = s_k
                    if r_i < l_k and r_k < l_j:
                        include = False
                        break
                if include:
                    # edge
                    e = (i, j)
                    edges.append(e)
                    # transition edge
                    T = transition_matrix[labels[i - 1], labels[j - 1]]
                    transition_edge[e] = T

    nodes = [i + 1 for i in range(len(spans))]

    start_node = nodes[0] - 1
    end_node = nodes[-1] + 1

    # add start/end nodes
    ext_edges = []
    for s in nodes:
        if s not in [k[1] for k in edges]:
            # start edge
            e = (start_node, s)  # add start node 0
            ext_edges.append(e)
            # transition edge
            T = transition_matrix[start_trans_index, labels[s - 1]]
            transition_edge[e] = T
        if s not in [k[0] for k in edges]:
            # end edge
            e = (s, end_node)
            ext_edges.append(e)
            # transition edge
            T = transition_matrix[labels[s - 1], end_trans_index]
            transition_edge[e] = T

    edges = edges + ext_edges
    # node scores
    node_scores = dict(zip(nodes, scores))

    # dict edge=>weight
    edge_weight = {}
    for e in edges:
        if e[1] == end_node:
            edge_weight[e] = transition_edge[e]
        else:
            edge_weight[e] = node_scores[e[1]] + transition_edge[e]  # edge_score(i->j) = node_score(j) + T_ij

    # dict node=>["previous nodes"]
    previous = defaultdict(list)
    for e in edges:
        previous[e[1]].append(e[0])
    previous[start_node] = []
    previous = dict(previous)

    nodes = [start_node] + nodes + [end_node]

    return nodes, edge_weight, previous


def compute_log_partition(nodes, edge_weight, previous, device="cpu"):
    c = torch.Tensor([(n != nodes[0]) * (-1e20) for n in nodes]).to(device)
    for k in nodes:
        for k_prev in previous[k]:  # previous nodes
            u = c[k].clone()
            v = c[k_prev].clone() + edge_weight[(k_prev, k)]
            c[k] = torch.logaddexp(input=u, other=v)
    return c[-1]


def compute_log_partition_stable_bis(nodes, edge_weight, previous, is_torch=True, device="cpu"):
    if is_torch:
        exp = torch.exp
        log = torch.log
    else:
        exp = np.exp
        log = np.log
    c = torch.Tensor([(n != nodes[0]) * (-1e20) for n in nodes]).to(device)
    for k in nodes:
        for k_prev in previous[k]:  # previous nodes
            u = c[k]
            v = c[k_prev] + edge_weight[(k_prev, k)]
            max_uv = max(u, v)
            c[k] = max_uv + log(exp(u - max_uv) + exp(v - max_uv))
    return c[-1]


class IntervalGraph(object):
    def __init__(self, spans, scores, labels=None, transition_matrix=None, is_sorted=False):
        assert len(spans) > 0
        if not is_sorted:
            idx_sorted = sorted(range(len(spans)), key=lambda x: spans[x][0])
            spans = [spans[i] for i in idx_sorted]
            scores = [scores[i] for i in idx_sorted]
            if labels is not None:
                labels = [labels[i] for i in idx_sorted]

        self.device = scores[0].device

        self.spans, self.scores = spans, scores
        # span => score
        self.span_score = dict(zip(spans, scores))

        if labels is None or transition_matrix is None:
            self.span_label = defaultdict(int)
            self.transition_matrix = defaultdict(int)
        else:
            self.span_label = dict(zip(spans, labels))
            self.transition_matrix = transition_matrix

        # create graph
        self.start_trans_index = -2
        self.end_trans_index = -1
        self.nodes, self.edge_weight, self.previous = create_graph(
            spans,
            scores,
            labels,
            transition_matrix,
            self.start_trans_index,
            self.end_trans_index
        )

        # partition
        self.log_Z = None

    def compute_log_Z(self):
        """compute the partition function Z"""
        if self.log_Z is None:
            self.log_Z = compute_log_partition(self.nodes, self.edge_weight, self.previous, device=self.device)
        return self.log_Z

    def compute_path_score(self, path):
        """compute score of a path"""
        score = 0
        prev_label = self.start_trans_index
        for span in path:
            sc = self.span_score[span]
            current_label = self.span_label[span]
            score += sc + self.transition_matrix[prev_label, current_label]
            prev_label = current_label
        # add end transition
        score += self.transition_matrix[prev_label, self.end_trans_index]
        return score

    def compute_path_prob(self, path):
        """compute probability of a path"""
        # score of path
        log_prob = self.compute_path_log_prob(path)
        return log_prob.exp()

    def compute_path_log_prob(self, path):
        """compute log probability of a path"""
        # score of path
        score = self.compute_path_score(path)
        # partition func
        log_Z = self.compute_log_Z()
        # compute log_prob
        log_prob = score - log_Z
        return log_prob

    def compute_path_nll(self, path):
        # negative log prob
        log_prob = self.compute_path_log_prob(path)
        return - log_prob

    def best_path(self): # sum of scores
        # compute best path for a given graph (nodes are span scores, edges are transition scores)
        G = nx.DiGraph()
        # negative score
        for (i, j), v in self.edge_weight.items():
            G.add_edge(i, j, weight=-v)
        best_path = nx.algorithms.bellman_ford_path(G, 0, len(self.spans) + 1, weight='weight')[1:-1]
        best_path = [self.spans[i - 1] for i in best_path]
        return best_path

    def enumerate_paths(self):
        """Enumerate all simple paths of the graph"""
        G = nx.DiGraph()
        for i, j in self.edge_weight.keys():
            G.add_edge(i, j)
        all_paths = [[self.spans[i - 1] for i in path[1:-1]]
                     for path in nx.all_simple_paths(G, 0, len(self.spans) + 1)]
        return all_paths

    def compute_log_Z_by_enumeration(self, all_paths=None):
        if all_paths is None:
            all_paths = self.enumerate_paths()
        all_scores = torch.stack(
            [self.compute_path_score(i) for i in all_paths]
        )
        return torch.logsumexp(all_scores, dim=0)

    def greedy_search(self):
        # Greedy decoding: select the highest scoring span at each step
        # See (Zaratiana et al., UM-IoS 2022) for details (https://aclanthology.org/2022.umios-1.1/)
        spans, scores = self.spans, self.scores
        # convert scores to list
        scores = [s.item() for s in scores]
        greedy_path = []
        span_score = list(zip(spans, scores))
        span_score = sorted(span_score, key=lambda x: -x[1])
        for i in range(len(span_score)):
            b = span_score[i]
            flag = False
            for new in greedy_path:
                if has_overlapping(b[0], new):
                    flag = True
                    break
            if not flag:
                greedy_path.append(b[0])
        new_list = sorted(greedy_path, key=lambda x: x[0])
        return new_list

    def exhaustive_search(self, scoring_func=np.mean):
        # See (Zaratiana et al., UM-IoS 2022) for details (https://aclanthology.org/2022.umios-1.1/)
        # enumerate all possible paths and compute their scores using the score function
        # return the path with the highest score
        all_paths = self.enumerate_paths()
        all_scores = [scoring_func([self.span_score[i].item() for i in path]) for path in all_paths]
        best_path = all_paths[np.argmax(all_scores)]
        return best_path

