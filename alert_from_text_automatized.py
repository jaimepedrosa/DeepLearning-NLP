import sys
import torch
import json
from pathlib import Path

# Imprimir el directorio actual y sys.path para depuración
print("Directorio actual:", Path.cwd())
print("sys.path inicial:", sys.path)

# Añadir las carpetas ner y sentimentAnalysis al sys.path
root = Path(__file__).resolve().parents[0]  # Subir un nivel desde alert_generation_automatized/ a p-final/
ner_path = str(root / "ner")
sentiment_path = str(root / "sentimentAnalysis")
alert_gen_path = str(root / "alert_generation_automatized")

# Imprimir las rutas que estamos intentando agregar
print("Agregando al sys.path:")
print(f"ner_path: {ner_path}")
print(f"sentiment_path: {sentiment_path}")
print(f"alert_gen_path: {alert_gen_path}")

sys.path.append(ner_path)
sys.path.append(sentiment_path)
sys.path.append(alert_gen_path)

# Imprimir sys.path después de agregar las rutas
print("sys.path después de agregar rutas:", sys.path)

from ner.evaluate import evaluate as evaluate_ner
from ner.preprocessing import load_model as load_ner_model
from sentimentAnalysis.model import SentimentLSTM
from sentimentAnalysis.embeddings import build_embeddings_matrix
from sentimentAnalysis.dataset import preprocess_text
from alert_generation_automatized.alert_generator import AlertGenerator

# Configuración de rutas
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

def parse_entities(pred_tags, sentence):
    """Extrae entidades de las etiquetas NER en formato BIO."""
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
        elif tag.startswith("I-") and ent_type == tag[2:].upper():
            current.append(tok)
        else:
            if current:
                entities.append({"text": " ".join(current), "type": ent_type})
            current = []
            ent_type = None
    if current:
        entities.append({"text": " ".join(current), "type": ent_type})
    return entities

def preprocess_sentiment_text(text, word_index, max_len=MAX_LEN):
    """Preprocesa el texto para el modelo de sentimiento."""
    text_clean = preprocess_text(text)
    tokens = text_clean.split()
    input_ids = [word_index.get(token, 0) for token in tokens][:max_len]
    padding_length = max_len - len(input_ids)
    input_ids += [0] * padding_length
    return input_ids

def predict_sentiment(sentence, model, word_index, device):
    """Predice el sentimiento de una oración usando SentimentLSTM."""
    model.eval()
    input_ids = preprocess_sentiment_text(sentence, word_index)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_label = torch.argmax(outputs, dim=1).item()
    
    return LABELS[pred_label]

def main():
    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading models...")

    # Cargar el modelo de alertas (usar distilgpt2 directamente)
    print("Using distilgpt2 for alert generation...")
    alert_model = AlertGenerator(model_name="distilgpt2", device=device)

    # Cargar el modelo de sentimiento
    with open(WORD_INDEX_PATH, 'r', encoding='utf-8') as f:
        word_index = json.load(f)
    embedding_matrix = build_embeddings_matrix(word_index, str(EMBEDDING_PATH))
    sentiment_model = SentimentLSTM(embedding_matrix, HIDDEN_DIM, NUM_CLASSES)
    sentiment_model.load_state_dict(torch.load(SENTIMENT_MODEL_PATH, map_location=device))
    sentiment_model.to(device)
    sentiment_model.eval()

    # Cargar el modelo NER
    ner_model = load_ner_model(NER_MODEL_PATH)
    ner_model.to(device)
    vocab = torch.load(VOCAB_PATH, map_location=device)
    tag_to_idx = torch.load(TAG_TO_IDX_PATH, map_location=device)
    idx_to_tag = torch.load(IDX_TO_TAG_PATH, map_location=device)
    ner_model.eval()

    # Frase de entrada
    sentence = input("Introduce una frase: ").strip()

    # Predecir NER
    ner_tags = evaluate_ner(ner_model, sentence, vocab, tag_to_idx, idx_to_tag, device)
    pred_tag_list = list(ner_tags.values())
    entities = parse_entities(pred_tag_list, sentence)

    # Seleccionar la primera entidad relevante
    entity = None
    for ent in entities:
        if ent["type"] in ["PER", "ORG", "LOC", "GPE", "GEO"]:
            entity = ent["text"]
            break
    if not entity:
        print("No se encontraron entidades relevantes en la oración.")
        return

    # Predecir el sentimiento
    sentiment = predict_sentiment(sentence, sentiment_model, word_index, device)

    # Generar la alerta
    prompt = f"Entity: {entity}, Sentiment: {sentiment}"
    generated_alert = alert_model.generate_alert(prompt)

    # Mostrar los resultados
    print("\n--- RESULTADOS ---")
    print(f"Frase: {sentence}")
    print(f"Etiquetas NER: {pred_tag_list}")
    print(f"Entidades: {entities}")
    print(f"Sentimiento: {sentiment}")
    print(f"Alerta generada: {generated_alert}")

if __name__ == "__main__":
    main()