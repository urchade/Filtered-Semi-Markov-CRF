from data import create_dataset
from model import NerSpanModel

path = "/Users/urchadezaratiana/Documents/remote-server/NER_datasets/CoNLL 2003"

train_dataset, dev_dataset, test_dataset, labels = create_dataset(path)

print("train_dataset[0]: ", train_dataset[0])

config_dict = {
    "hidden_size": 768,
    "entity_types": labels,
    "max_width": 10,
    "model_name": "bert-base-cased",
    "fine_tune": True,
    "subtoken_pooling": "first",
    "span_mode": "endpoints",
    "dropout": 0.1,
    "sample_rate": 0.1,
    "model_type": "gss",
    "decoding": "best"
}



# convert dict to namespace
from argparse import Namespace
config_dict = Namespace(**config_dict)

model = NerSpanModel(config_dict)

loader = model.create_dataloader([i.values() for i in train_dataset], batch_size=2, shuffle=True)

x = next(iter(loader))

out = model.compute_loss(x)

#preds = model.predict(x)

#print("out: ", out)
#print("preds: ", preds)