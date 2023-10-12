import torch
from .metric import compute_prf


class Evaluator:
    def __init__(self, all_true, all_outs):
        self.all_true = all_true
        self.all_outs = all_outs

    def get_entities_fr(self, ents, with_type=True):
        all_ents = []
        for s, e, lab in ents:
            all_ents.append([lab, (s, e)])
        return all_ents

    def transform_data(self, with_type=True):
        all_true_ent = []
        all_outs_ent = []
        for i, j in zip(self.all_true, self.all_outs):
            e = self.get_entities_fr(i, with_type=with_type)
            all_true_ent.append(e)
            e = self.get_entities_fr(j, with_type=with_type)
            all_outs_ent.append(e)
        return all_true_ent, all_outs_ent

    @torch.no_grad()
    def evaluate(self):
        output = {}
        all_true_typed, all_outs_typed = self.transform_data(with_type=True)
        output["r"] = compute_prf(all_true_typed, all_outs_typed)
        precision, recall, f1 = compute_prf(all_true_typed, all_outs_typed).values()
        output_str = f"P: {precision:.2%}\tR: {recall:.2%}\tF1: {f1:.2%}\n"
        return output_str, f1
