import numpy as np
from tqdm import tqdm
from collections import Counter
import pandas as pd
import re
import json


def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower()


def build_vocab_from_dataset(file_path):
    df = pd.read_csv(file_path)
    all_text = ' '.join(df['text'].apply(preprocess_text).tolist())
    tokens = all_text.split()
    counter = Counter(tokens)
    word_index = {word: idx + 1 for idx, (word, _) in enumerate(counter.items())}  # idx+1 para padding token
    return word_index


def check_embedding_coverage(word_index, embedding_path):
    embeddings_index = {}
    with open(embedding_path, encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    covered = {}
    oov = {}

    for word in word_index:
        if word in embeddings_index:
            covered[word] = embeddings_index[word]
        else:
            oov[word] = None

    vocab_size = len(word_index)
    covered_size = len(covered)
    coverage_percent = covered_size / vocab_size * 100

    print(f"Vocabulary size: {vocab_size}")
    print(f"Covered by embeddings: {covered_size}")
    print(f"Coverage: {coverage_percent:.2f}%")

    print("\nExamples of out-of-vocabulary words:")
    print(list(oov.keys())[:20])

    return covered, oov


def save_vocab(word_index, output_path='word_index.json'):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(word_index, f, ensure_ascii=False, indent=4)
    print(f"Vocabulary saved to {output_path}")


if __name__ == '__main__':
    FILE_PATH = 'csvs/ner_with_sentiment.csv'
    EMBEDDING_PATH = 'embeds/glove.6B.300d.txt'

    word_index = build_vocab_from_dataset(FILE_PATH)
    check_embedding_coverage(word_index, EMBEDDING_PATH)
    save_vocab(word_index, 'word_index.json')


"""
Loading embeddings: 400000it [00:12, 32858.16it/s]
Vocabulary size: 31294
Covered by embeddings: 26885
Coverage: 85.91%

Examples of out-of-vocabulary words:
['lowlevel', 'oilrich', 'royaldutch', 'sixparty', 'majliseamal', '6834', 'sixyear', 'faridullah', 'alqaida', 'nancyamelia', 'onestop', 'islamistcontrolled', 'leonella', 'sgorbati', 'natoled', 'mazaresharif', 'itartass', '27year', 'internationallyrecognized', 'usbrokered']
Vocabulary saved to word_index.json
"""