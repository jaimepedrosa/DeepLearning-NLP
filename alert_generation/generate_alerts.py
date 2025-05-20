import pandas as pd
import ast
from rules import generate_alert

# Convierte predicción numérica a string de clase
def normalize_sentiment(sent):
    mapping = {-1: "neutral", 0: "negative", 1: "positive"}
    if isinstance(sent, str):
        return sent.lower()
    return mapping.get(sent, "neutral")

def parse_entities(tag_str, token_str):
    tags = ast.literal_eval(tag_str)
    tokens = token_str.split()
    entities = []
    current = []
    ent_type = None

    for tok, tag in zip(tokens, tags):
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

def run_alert_generation(input_path, output_path):
    df = pd.read_csv(input_path)
    alerts = []

    for _, row in df.iterrows():
        sentiment = normalize_sentiment(row["sentiment_predicted"])
        entities = parse_entities(row["NER Tags"], row["Sentence"])
        alert = generate_alert(entities, sentiment, row["Sentence"])
        alerts.append(alert if alert else "")

    df["Alert"] = alerts
    df.to_csv(output_path, index=False)
    print(f"Alerts written to {output_path}")

if __name__ == "__main__":
    run_alert_generation("alert_generation/outputs/merged_train.csv", "alert_generation/outputs/alerts_train.csv")
    run_alert_generation("alert_generation/outputs/merged_val.csv", "alert_generation/outputs/alerts_val.csv")
