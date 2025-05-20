def predict_alert(model, prompt):
    """Genera una alerta a partir de un prompt."""
    return model.generate_alert(prompt)