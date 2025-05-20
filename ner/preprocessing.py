import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast


def load_data(file_path):
    df = pd.read_csv(file_path)
    sentences = df['Sentence'].tolist()
    ner_tags = [ast.literal_eval(tags) for tags in df['NER Tag'].tolist()]
    return sentences, ner_tags


def build_tag_map(ner_tags):
    unique_tags = set(tag for tags in ner_tags for tag in tags)
    tag_to_idx = {'<PAD>': 0}
    for tag in unique_tags:
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    return tag_to_idx, idx_to_tag


def build_vocab(sentences):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def load_model(model_path):
    model = torch.load(model_path)
    return model