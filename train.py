import argparse

import torch
import yaml

from data import create_dataset
from model import NerSpanModel
from train_utils import train_model


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--config", type=str, default="configs/conll.yaml", help="Path to config file")
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the log directory')
    return parser


def load_config_as_namespace(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)

if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = parser.parse_args()

    # load config
    config = load_config_as_namespace(args.config)

    # overwrite config log_dir
    config.log_dir = args.log_dir

    # load dataset
    train_dataset, dev_dataset, test_dataset, labels = create_dataset(config.dataset_path)

    # overwrite config entity_types
    config.entity_types = labels

    # create model
    model = NerSpanModel(config)

    # optimizer
    optimizer = model.get_optimizer(config)

    # get hyperparameters
    train_bs, val_bs = config.batch_size
    num_epochs, num_steps, warmup_ratio, eval_steps = config.training[0].values()

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    # train
    train_model(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        min_num_steps=num_steps,
        num_epochs=num_epochs,
        eval_every=eval_steps,
        log_dir=config.log_dir,
        warmup_ratio=warmup_ratio,
        train_batch_size=train_bs,
        val_batch_size=train_bs,
        device=device
    )
