# DL-NP-Final-Project

Link a los embeddings preentrenados para el Sentiment Analysis
https://drive.google.com/file/d/13dcTSIfoyhFg6dHp2H13fn1Ubefj6x7O/view?usp=drive_link

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch (`torch`)
- Transformers (`transformers`)
- Pandas (`pandas`)
- NumPy (`numpy`)
- Scikit-learn (`scikit-learn`)

Install the required packages using pip:
```bash
pip install torch transformers pandas numpy scikit-learn
```

## Trying the Model

To test the model yourself, you have two options depending on the alert generation approach you prefer. 

### Option 1: Rule-Based Alerts
For the version using hardcoded rules, open a terminal and run the following command:
```bash
python3 alert_from_text.py
```

### Option 2: Automatized Generated Alerts
For the version using a LLM model, such as (`distilgpt2`) open a terminal and run the following command:
```bash
python3 alert_from_text_automatized.py
````

Then, in both of them, you just have to write an input to try them out.

You may want to take into account that the vocabulary used to train the model was mainly about geopolithics, so maybe your favorite footballer player may not be recognized.

