import torch
from torch.utils.data import DataLoader
from dataset import prepare_data
from embeddings import build_embeddings_matrix
from model import SentimentLSTM
from train import train_model
import json

# Config
FILE_PATH = 'ner_with_sentiment.csv'
EMBEDDING_PATH = 'embeds/glove.6B.300d.txt'
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10
HIDDEN_DIM = 128
NUM_CLASSES = 3

# Cargar vocabulario
with open('word_index.json', 'r', encoding='utf-8') as f:
    word_index = json.load(f)

# Data
train_dataset, val_dataset = prepare_data(FILE_PATH, word_index, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Embeddings
embedding_matrix = build_embeddings_matrix(word_index, EMBEDDING_PATH)

# Model
model = SentimentLSTM(embedding_matrix, HIDDEN_DIM, NUM_CLASSES)

# Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train
train_model(model, train_loader, val_loader, EPOCHS, optimizer, criterion, device)

print("Training completed!")

# Save model
torch.save(model.state_dict(), 'sentiment_model.pt')
print("Model saved to sentiment_model.pt")

"""
Loading embeddings: 400000it [00:12, 31879.29it/s]
Training Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████| 1199/1199 [01:57<00:00, 10.24it/s]
Epoch 1/10, Loss: 0.5932
              precision    recall  f1-score   support

    negative       0.78      0.80      0.79      3369
     neutral       0.84      0.86      0.85      5750
    positive       0.62      0.31      0.41       473

    accuracy                           0.81      9592
   macro avg       0.74      0.66      0.68      9592
weighted avg       0.80      0.81      0.80      9592

Training Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████| 1199/1199 [02:06<00:00,  9.51it/s]
Epoch 2/10, Loss: 0.4199
              precision    recall  f1-score   support

    negative       0.80      0.83      0.81      3369
     neutral       0.85      0.87      0.86      5750
    positive       0.71      0.36      0.48       473

    accuracy                           0.83      9592
   macro avg       0.79      0.69      0.72      9592
weighted avg       0.83      0.83      0.83      9592

Training Epoch 3: 100%|█████████████████████████████████████████████████████████████████████████████| 1199/1199 [01:56<00:00, 10.28it/s]
Epoch 3/10, Loss: 0.3619
              precision    recall  f1-score   support

    negative       0.83      0.83      0.83      3369
     neutral       0.86      0.89      0.87      5750
    positive       0.69      0.48      0.56       473

    accuracy                           0.84      9592
   macro avg       0.79      0.73      0.76      9592
weighted avg       0.84      0.84      0.84      9592

Training Epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████| 1199/1199 [01:54<00:00, 10.48it/s]
Epoch 4/10, Loss: 0.3109
              precision    recall  f1-score   support

    negative       0.81      0.85      0.83      3369
     neutral       0.88      0.86      0.87      5750
    positive       0.66      0.56      0.60       473

    accuracy                           0.84      9592
   macro avg       0.78      0.76      0.77      9592
weighted avg       0.84      0.84      0.84      9592

Training Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████| 1199/1199 [01:49<00:00, 10.91it/s]
Epoch 5/10, Loss: 0.2575
              precision    recall  f1-score   support

    negative       0.86      0.80      0.83      3369
     neutral       0.86      0.90      0.88      5750
    positive       0.67      0.57      0.61       473

    accuracy                           0.85      9592
   macro avg       0.79      0.76      0.77      9592
weighted avg       0.85      0.85      0.85      9592

Training Epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████| 1199/1199 [01:54<00:00, 10.43it/s]
Epoch 6/10, Loss: 0.2030
              precision    recall  f1-score   support

    negative       0.83      0.83      0.83      3369
     neutral       0.87      0.88      0.87      5750
    positive       0.64      0.51      0.57       473

    accuracy                           0.85      9592
   macro avg       0.78      0.74      0.76      9592
weighted avg       0.84      0.85      0.84      9592

Training Epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████| 1199/1199 [01:57<00:00, 10.19it/s]
Epoch 7/10, Loss: 0.1558
              precision    recall  f1-score   support

    negative       0.83      0.82      0.83      3369
     neutral       0.87      0.87      0.87      5750
    positive       0.61      0.62      0.61       473

    accuracy                           0.84      9592
   macro avg       0.77      0.77      0.77      9592
weighted avg       0.84      0.84      0.84      9592

Training Epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████| 1199/1199 [01:53<00:00, 10.54it/s]
Epoch 8/10, Loss: 0.1135
              precision    recall  f1-score   support

    negative       0.84      0.80      0.82      3369
     neutral       0.86      0.89      0.87      5750
    positive       0.64      0.58      0.61       473

    accuracy                           0.84      9592
   macro avg       0.78      0.76      0.77      9592
weighted avg       0.84      0.84      0.84      9592

Training Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████| 1199/1199 [01:52<00:00, 10.62it/s]
Epoch 9/10, Loss: 0.0875
              precision    recall  f1-score   support

    negative       0.83      0.83      0.83      3369
     neutral       0.88      0.87      0.87      5750
    positive       0.59      0.66      0.63       473

    accuracy                           0.84      9592
   macro avg       0.77      0.79      0.78      9592
weighted avg       0.85      0.84      0.84      9592

Training Epoch 10: 100%|████████████████████████████████████████████████████████████████████████████| 1199/1199 [02:08<00:00,  9.32it/s]
Epoch 10/10, Loss: 0.0671
              precision    recall  f1-score   support

    negative       0.78      0.86      0.82      3369
     neutral       0.88      0.84      0.86      5750
    positive       0.65      0.55      0.60       473

    accuracy                           0.83      9592
   macro avg       0.77      0.75      0.76      9592
weighted avg       0.83      0.83      0.83      9592

Training completed!
Model saved to sentiment_model.pt

"""