import pandas as pd

def extract_entities(ner_tags):
    """Extrae entidades de las etiquetas NER en formato BIO."""
    entities = []
    current_entity = []
    current_type = None

    for i, tag in enumerate(ner_tags):
        if tag.startswith("B-"):
            if current_entity:  # Guardar la entidad anterior si existe
                entities.append((current_type, " ".join(current_entity)))
            current_entity = [str(ner_tags[i]).split()[-1]]  # Obtener la palabra asociada
            current_type = tag[2:]  # Tipo de entidad (e.g., "per", "org")
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_entity.append(str(ner_tags[i]).split()[-1])
        else:
            if current_entity:
                entities.append((current_type, " ".join(current_entity)))
            current_entity = []
            current_type = None

    if current_entity:  # Guardar la última entidad
        entities.append((current_type, " ".join(current_entity)))
    return entities

def prepare_alert_data(csv_path):
    df = pd.read_csv(csv_path)
    prompts = []
    alerts = []

    for _, row in df.iterrows():
        ner_tags = eval(row["NER Tags"])
        sentiment = row["sentiment_predicted"]

        entities = extract_entities(ner_tags)
        if not entities:
            continue

        entity = None
        for ent_type, ent_name in entities:
            if ent_type in ["per", "org", "loc", "gpe", "geo"]:
                entity = ent_name
                break
        if not entity:
            continue

        prompt = f"Generate alert: Entity: {entity}, Sentiment: {sentiment}"  # Agregar instrucción
        prompts.append(prompt)

        if sentiment == "negative":
            alert = f"Reputation risk: {entity} mentioned negatively"
        elif sentiment == "neutral":
            alert = f"Neutral mention: {entity} referenced in a neutral context"
        else:
            alert = f"Positive spotlight: {entity} received positive mention"
        alerts.append(alert)

    return prompts, alerts


def load_alert_data(train_csv, val_csv):
    """Carga los datos de entrenamiento y validación."""
    train_prompts, train_alerts = prepare_alert_data(train_csv)
    val_prompts, val_alerts = prepare_alert_data(val_csv)
    return train_prompts, train_alerts, val_prompts, val_alerts

def extract_entities(ner_tags):
    """Extrae entidades de las etiquetas NER en formato BIO."""
    entities = []
    current_entity = []
    current_type = None

    for i, tag in enumerate(ner_tags):
        if tag.startswith("B-"):
            if current_entity:
                entities.append((current_type, " ".join(current_entity)))
            current_entity = [str(ner_tags[i]).split()[-1]]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_entity.append(str(ner_tags[i]).split()[-1])
        else:
            if current_entity:
                entities.append((current_type, " ".join(current_entity)))
            current_entity = []
            current_type = None

    if current_entity:
        entities.append((current_type, " ".join(current_entity)))
    return entities