import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import re


def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower()


def prepare_data(file_path, word_index, max_len=100):
    df = pd.read_csv(file_path)
    df['text'] = df['Sentence'].apply(preprocess_text)


    df['sentiment_encoded'] = df['Sentiment'].apply(lambda x: 1 if x == -1 else (2 if x == 1 else 0)).astype(int) # negative de 0 a 0, neutral de -1 a 1, positivo de 1 a 2

    def encode_text(text):
        tokens = text.split()
        input_ids = [word_index.get(token, 0) for token in tokens][:max_len]
        padding_length = max_len - len(input_ids)
        input_ids += [0] * padding_length
        return input_ids

    df['input_ids'] = df['text'].apply(encode_text)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['input_ids'].tolist(),
        df['sentiment_encoded'].tolist(),
        test_size=0.2,
        random_state=42
    )

    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {
                'input_ids': torch.tensor(self.texts[idx], dtype=torch.long),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    train_dataset = SentimentDataset(train_texts, train_labels)
    val_dataset = SentimentDataset(val_texts, val_labels)

    return train_dataset, val_dataset