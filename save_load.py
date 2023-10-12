import torch

from model import NerSpanModel


def save_model(current_model, path):
    model_args = current_model.config
    dict_save = {"model_weights": current_model.state_dict(), "model_args": model_args}
    torch.save(dict_save, path)


def load_model(path, model_name=None):
    dict_load = torch.load(path, map_location=torch.device('cpu'))
    model_args = dict_load["model_args"]

    if model_name is not None:
        model_args["model_name"] = model_name

    loaded_model = NerSpanModel(model_args)
    loaded_model.load_state_dict(dict_load["model_weights"])
    return loaded_model
