from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch

def train_alert_generator(model, train_prompts, train_alerts, val_prompts, val_alerts):
    # Entrena el modelo de generación de alertas.
    # Preparar los datos como un dataset de Hugging Face
    train_data = {"input": train_prompts, "output": train_alerts}
    val_data = {"input": val_prompts, "output": val_alerts}
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    # Tokenizar los datos
    train_encodings = model.tokenize_data(train_prompts, train_alerts)
    val_encodings = model.tokenize_data(val_prompts, val_alerts)

    # Convertir a datasets
    train_dataset = train_dataset.add_column("input_ids", train_encodings["input_ids"].tolist())
    train_dataset = train_dataset.add_column("attention_mask", train_encodings["attention_mask"].tolist())
    train_dataset = train_dataset.add_column("labels", train_encodings["labels"].tolist())

    val_dataset = val_dataset.add_column("input_ids", val_encodings["input_ids"].tolist())
    val_dataset = val_dataset.add_column("attention_mask", val_encodings["attention_mask"].tolist())
    val_dataset = val_dataset.add_column("labels", val_encodings["labels"].tolist())

    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./alert_generator",
        num_train_epochs=10,  # Reducido para CPU
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=200,
        fp16=False,  # No compatible con CPU
        gradient_accumulation_steps=2,
        eval_strategy="steps",  # Cambiado de evaluation_strategy a eval_strategy
        eval_steps=500,
        learning_rate=5e-5,
    )

    # Crear el Trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Entrenar
    trainer.train()
    return model