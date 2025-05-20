import numpy as np
import torch
from tqdm import tqdm


def build_embeddings_matrix(word_index, embedding_path, embedding_dim=300):
    embeddings_index = {}
    with open(embedding_path, encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    return torch.tensor(embedding_matrix, dtype=torch.float)