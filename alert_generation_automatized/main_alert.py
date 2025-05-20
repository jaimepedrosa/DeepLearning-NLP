from alert_preprocessing import load_alert_data, extract_entities
from alert_generator import AlertGenerator
from train_alert import train_alert_generator
from predict_alert import predict_alert
import pandas as pd

def main():
    # Rutas a los CSVs
    train_csv = "data/merged_train.csv"
    val_csv = "data/merged_val.csv"

    # Cargar los datos
    train_prompts, train_alerts, val_prompts, val_alerts = load_alert_data(train_csv, val_csv)

    # Inicializar el modelo de generación de alertas
    device = "cpu"
    alert_model = AlertGenerator(device=device)

    # Entrenar el modelo
    alert_model = train_alert_generator(alert_model, train_prompts, train_alerts, val_prompts, val_alerts)

    # Cargar los CSVs originales para obtener NER Tags y Sentiment
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Generar alertas para todas las filas del dataset de entrenamiento
    train_results = []
    for _, row in train_df.iterrows():
        ner_tags = eval(row["NER Tags"])  # Convertir de string a lista
        sentiment = row["sentiment_predicted"]

        # Extraer entidades usando la función existente
        entities = extract_entities(ner_tags)
        entity = None
        for ent_type, ent_name in entities:
            if ent_type in ["per", "org", "loc", "gpe", "geo"]:
                entity = ent_name
                break
        if not entity:
            continue

        # Crear el prompt
        prompt = f"Entity: {entity}, Sentiment: {sentiment}"
        # Generar la alerta
        generated_alert = predict_alert(alert_model, prompt)
        train_results.append({
            "Sentence": row["Sentence"],
            "NER Tags": row["NER Tags"],
            "Sentiment": sentiment,
            "Prompt": prompt,
            "Generated Alert": generated_alert
        })

    # Generar alertas para todas las filas del dataset de validación
    val_results = []
    for _, row in val_df.iterrows():
        ner_tags = eval(row["NER Tags"])
        sentiment = row["sentiment_predicted"]

        entities = extract_entities(ner_tags)
        entity = None
        for ent_type, ent_name in entities:
            if ent_type in ["per", "org", "loc", "gpe", "geo"]:
                entity = ent_name
                break
        if not entity:
            continue

        prompt = f"Entity: {entity}, Sentiment: {sentiment}"
        generated_alert = predict_alert(alert_model, prompt)
        val_results.append({
            "Sentence": row["Sentence"],
            "NER Tags": row["NER Tags"],
            "Sentiment": sentiment,
            "Prompt": prompt,
            "Generated Alert": generated_alert
        })

    # Guardar los resultados en CSVs
    train_results_df = pd.DataFrame(train_results)
    val_results_df = pd.DataFrame(val_results)
    train_results_df.to_csv("predicted/train_alert_predictions.csv", index=False)
    val_results_df.to_csv("predicted/val_alert_predictions.csv", index=False)
    print("Predicciones guardadas en 'train_alert_predictions.csv' y 'val_alert_predictions.csv'")

if __name__ == "__main__":
    main()