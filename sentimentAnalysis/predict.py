import torch
from model import SentimentLSTM
from embeddings import build_embeddings_matrix
from dataset import preprocess_text
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Config
EMBEDDING_PATH = 'embeds/glove.6B.300d.txt'
MODEL_PATH = 'model/sentiment_model.pt'
WORD_INDEX_PATH = 'word_index.json'
MAX_LEN = 100
HIDDEN_DIM = 128
NUM_CLASSES = 3
FILE_PATH = 'ner_with_sentiment.csv'
BATCH_SIZE = 512  # <--- batch size para evitar romper la RAM

# Labels
LABELS = ['negative', 'neutral', 'positive']

# Cargar vocabulario
with open(WORD_INDEX_PATH, 'r', encoding='utf-8') as f:
    word_index = json.load(f)

# Cargar embeddings
embedding_matrix = build_embeddings_matrix(word_index, EMBEDDING_PATH)

# Cargar modelo
model = SentimentLSTM(embedding_matrix, HIDDEN_DIM, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Función de preprocesado de textos
def preprocess_texts(texts):
    input_ids_batch = []
    for text in texts:
        text_clean = preprocess_text(text)
        tokens = text_clean.split()
        input_ids = [word_index.get(token, 0) for token in tokens][:MAX_LEN]
        padding_length = MAX_LEN - len(input_ids)
        input_ids += [0] * padding_length
        input_ids_batch.append(input_ids)
    return input_ids_batch

# Función de predicción por batches
def predict_batch(texts):
    input_ids_all = preprocess_texts(texts)
    predictions = []

    for i in tqdm(range(0, len(input_ids_all), BATCH_SIZE), desc="Predicting"):
        batch_input_ids = input_ids_all[i:i + BATCH_SIZE]
        input_tensor = torch.tensor(batch_input_ids, dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_labels = torch.argmax(probs, dim=1).tolist()
            predictions.extend([LABELS[pred] for pred in pred_labels])

    return predictions

if __name__ == '__main__':
    # Leer dataset original
    df = pd.read_csv(FILE_PATH)

    # Split train / val / test igual que en el entrenamiento
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Para cada partición, predecimos y guardamos el CSV
    for split_name, split_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        print(f'Procesando {split_name}...')

        # Generar predicciones en batches
        split_df['sentiment_predicted'] = predict_batch(split_df['Sentence'])

        # Guardar a CSV
        output_file = f'{split_name}_predictions.csv'
        split_df.to_csv(output_file, index=False)
        print(f'Saved predictions to {output_file}')

    print("Todas las predicciones guardadas correctamente.")
