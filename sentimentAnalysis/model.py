import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_classes):
        super(SentimentLSTM, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True) # ner si o si, SA comprobar si es necesario bidireccional
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(64, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        x = F.relu(self.fc(pooled))
        x = self.dropout(x)
        output = self.output(x)
        return output


# mirar word2vec, fast_text, glove, exploracion de embeddings para ver cual tiene mejor solapamiento con el vocabulario del training
# mirar embedding dinamico