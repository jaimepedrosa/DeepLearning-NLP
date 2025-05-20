import torch


class BiLSTMNER(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, dropout=0.1):
        super(BiLSTMNER, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=1, 
                           bidirectional=True, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden_dim * 2, tagset_size) 

    def forward(self, sentence):
        embeds = self.embedding(sentence)  
        lstm_out, _ = self.lstm(embeds)   
        lstm_out = self.dropout(lstm_out)
        tag_space = self.linear(lstm_out)  
        return tag_space