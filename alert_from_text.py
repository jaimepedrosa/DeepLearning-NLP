import torch
import json
import ast
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[0]
sys.path.append(str(root / "ner"))
sys.path.append(str(root / "sentimentAnalysis"))
sys.path.append(str(root / "alert_generation"))


from sentimentAnalysis.model import SentimentLSTM
from sentimentAnalysis.embeddings import build_embeddings_matrix
from sentimentAnalysis.dataset import preprocess_text

from ner.model import BiLSTMNER
from ner.evaluate import evaluate as evaluate_ner

from alert_generation.rules import generate_alert


EMBEDDING_PATH = root / 'sentimentAnalysis' / 'embeds' / 'glove.6B.300d.txt'
SENTIMENT_MODEL_PATH = root / 'sentimentAnalysis' / 'model' / 'sentiment_model.pt'
NER_MODEL_PATH = root / 'ner' / 'models' / 'ner_model.pt'
VOCAB_PATH = root / 'ner' / 'models' / 'vocab.pt'
TAG_TO_IDX_PATH = root / 'ner' / 'models' / 'tag_to_idx.pt'
IDX_TO_TAG_PATH = root / 'ner' / 'models' / 'idx_to_tag.pt'
WORD_INDEX_PATH = root / 'sentimentAnalysis' / 'word_index.json'

MAX_LEN = 100
HIDDEN_DIM = 128
NUM_CLASSES = 3
LABELS = ['negative', 'neutral', 'positive']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def predict_sentiment(sentence, word_index, model):
    tokens = preprocess_text(sentence).split()
    input_ids = [word_index.get(token, 0) for token in tokens][:MAX_LEN]
    input_ids += [0] * (MAX_LEN - len(input_ids))
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1).item()
    return LABELS[pred]

def parse_entities(pred_tags, sentence):
    tokens = sentence.split()
    entities = []
    current = []
    ent_type = None

    for tok, tag in zip(tokens, pred_tags):
        if tag.startswith("B-"):
            if current:
                entities.append({"text": " ".join(current), "type": ent_type})
            current = [tok]
            ent_type = tag[2:].upper()
        elif tag.startswith("I-") and current:
            current.append(tok)
        else:
            if current:
                entities.append({"text": " ".join(current), "type": ent_type})
                current = []
    if current:
        entities.append({"text": " ".join(current), "type": ent_type})
    return entities



def main():
    print("Loading models...")

    with open(WORD_INDEX_PATH, 'r', encoding='utf-8') as f:
        word_index = json.load(f)
    embedding_matrix = build_embeddings_matrix(word_index, str(EMBEDDING_PATH))
    sentiment_model = SentimentLSTM(embedding_matrix, HIDDEN_DIM, NUM_CLASSES)
    sentiment_model.load_state_dict(torch.load(SENTIMENT_MODEL_PATH, map_location=device))
    sentiment_model.eval()

    vocab = torch.load(VOCAB_PATH)
    tag_to_idx = torch.load(TAG_TO_IDX_PATH)
    idx_to_tag = torch.load(IDX_TO_TAG_PATH)

    ner_model = BiLSTMNER(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=128,
        tagset_size=len(tag_to_idx)
    ).to(device)
    ner_model = torch.load(NER_MODEL_PATH, map_location=device)
    ner_model.to(device)
    ner_model.eval()

    # Frase de entrada
    sentence = input("Introduce una frase: ").strip()

    # Predicciones
    predicted_sentiment = predict_sentiment(sentence, word_index, sentiment_model)
    ner_tags = evaluate_ner(ner_model, sentence, vocab, tag_to_idx, idx_to_tag, device)
    pred_tag_list = list(ner_tags.values())
    entities = parse_entities(pred_tag_list, sentence)
    alert = generate_alert(entities, predicted_sentiment, sentence)

    # Salida
    print("\n--- RESULTADOS ---")
    print(f"Frase: {sentence}")
    print(f"Sentimiento: {predicted_sentiment}")
    print(f"Entidades: {entities}")
    print(f"Alerta generada: {alert if alert else 'No se ha generado ninguna alerta.'}")


if __name__ == "__main__":
    main()
