import os
import glob
import json


def load_json_from_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def open_content(path):
    paths = glob.glob(os.path.join(path, "*.json"))
    datasets = {'train': None, 'dev': None, 'test': None, 'labels': None}

    for p in paths:
        for key in datasets.keys():
            if key in p:
                datasets[key] = load_json_from_file(p)

    return datasets['train'], datasets['dev'], datasets['test'], datasets['labels']


def get_word_positions(start_char, end_char, words):
    start_word, end_word, char_count = None, None, 0

    for i, word in enumerate(words):
        if char_count == start_char:
            start_word = i
        if char_count + len(word) == end_char:
            end_word = i
            break
        char_count += len(word) + 1

    return start_word, end_word


def process(data):
    words = data['sentence'].split()
    entities = []

    for entity in data['entities']:
        start_char, end_char = entity['pos']
        start_word, end_word = get_word_positions(start_char, end_char, words)
        entities.append((start_word, end_word, entity['type']))

    return {"tokens": words, "ner": entities}


def create_dataset(path):
    train, dev, test, labels = open_content(path)
    train_dataset = [process(data) for data in train]
    dev_dataset = [process(data) for data in dev]
    test_dataset = [process(data) for data in test]

    return train_dataset, dev_dataset, test_dataset, labels