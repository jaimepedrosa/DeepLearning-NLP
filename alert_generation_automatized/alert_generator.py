import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class AlertGenerator:
    def __init__(self, model_name="distilgpt2", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Configurar el pad_token usando el eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Cargar el modelo sin configuraciones avanzadas
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            low_cpu_mem_usage=False,
        ).to(device)
        self.device = device

    def tokenize_data(self, prompts, alerts=None, max_length=50):
        """Tokeniza los prompts y alertas para entrenamiento o inferencia."""
        encodings = self.tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        if alerts:
            labels = self.tokenizer(
                alerts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )["input_ids"]
            encodings["labels"] = labels

        return encodings

    def generate_alert(self, prompt, max_length=50):
        """Genera una alerta a partir de un prompt."""
        self.model.eval()
        # Modificar el prompt para incluir una instrucción
        modified_prompt = f"Generate alert: {prompt}"
        inputs = self.tokenizer(modified_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,  # Evitar repetición de bigramas
            top_p=0.95,  # Usar muestreo top-p para mayor diversidad
            temperature=0.7,  # Reducir la aleatoriedad
        )
        alert = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Limpiar la salida para eliminar el prompt inicial si aparece
        if alert.startswith("Generate alert: "):
            alert = alert[len("Generate alert: "):]
        return alert.strip()