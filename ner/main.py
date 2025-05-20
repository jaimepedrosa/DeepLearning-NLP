import torch
from ner.model import BiLSTMNER
from dataset import get_dataloaders
from train import train
from evaluate import evaluate
from preprocessing import load_model, build_vocab, build_tag_map
import pandas as pd 
import os 


file = "data/ner_with_sentiment.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    batch_size = 2
    embedding_dim = 100
    hidden_dim = 128
    dropout = 0.2
    epochs = 20


    train_loader, val_loader, vocab, tag_to_idx, idx_to_tag = get_dataloaders(file, batch_size)

    print(f"Vocabulary Size: {len(vocab)}, Tag Size: {len(tag_to_idx)}")
    print(f"Tag to Index: {tag_to_idx}")
    model = BiLSTMNER(len(vocab), embedding_dim, hidden_dim, len(tag_to_idx), dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    train(model, train_loader, val_loader, optimizer, loss, epochs, device)
    torch.save(model, "ner/models/ner_model.pt")
    torch.save(vocab, "ner/models/vocab.pt")
    torch.save(tag_to_idx, "ner/models/tag_to_idx.pt")
    torch.save(idx_to_tag, "ner/models/idx_to_tag.pt")

    return vocab, tag_to_idx, idx_to_tag, device


def predict():
    
    model = load_model("ner/models/ner_model.pt")
    model.to(device)
    vocab = torch.load("ner/models/vocab.pt")
    tag_to_idx = torch.load("ner/models/tag_to_idx.pt")
    idx_to_tag = torch.load("ner/models/idx_to_tag.pt")
    

    sentence = "Google announced a new product in New York on Monday ."
    result = evaluate(model, sentence, vocab, tag_to_idx, idx_to_tag, device)
    print("Predicciones:")
    for word, tag in result.items():
        print(f"{word}: {tag}")

def predict():
    model = load_model("ner/models/ner_model.pt")
    model.to(device)
    vocab = torch.load("ner/models/vocab.pt")
    tag_to_idx = torch.load("ner/models/tag_to_idx.pt")
    idx_to_tag = torch.load("ner/models/idx_to_tag.pt")
    
    sentence = "The Eiffel Tower is located in Paris , France ."
    result = evaluate(model, sentence, vocab, tag_to_idx, idx_to_tag, device)
    print("Predicciones:")
    for word, tag in result.items():
        print(f"{word}: {tag}")

def get_results():
    train_loader, val_loader, vocab, _, _ = get_dataloaders(file, 1)
    model = load_model("ner/models/ner_model.pt")
    model.to(device)
    model.eval()
    tag_to_idx = torch.load("ner/models/tag_to_idx.pt")
    idx_to_tag = torch.load("ner/models/idx_to_tag.pt")

    train_sentences = []
    val_sentences = []
    train_pred_tags = []
    val_pred_tags = []

    idx_to_word = {idx: word for word, idx in vocab.items()}
    os.makedirs("ner/results", exist_ok=True)

    with torch.no_grad():
        i = 0
        for idx, (sentences, _) in enumerate(train_loader):
            i += 1
            if i % 200 == 0:
                print(f"Procesando train batch {i}")
            sentences = sentences.to(device)
            outputs = model(sentences)  # [1, seq_len, tagset_size]
            preds = torch.argmax(outputs, dim=-1)  # [1, seq_len]
            pred_tags = [idx_to_tag[idx.item()] for idx in preds.squeeze(0)]
            train_pred_tags.append(pred_tags)
            
            # Convertir tensor de entrada a oración legible
            sentence_words = [idx_to_word.get(idx.item(), '<UNK>') for idx in sentences.squeeze(0).cpu()]
            sentence = " ".join(word for word in sentence_words if word != '<PAD>')
            train_sentences.append(sentence)

    print("Training completed")
    train_df = pd.DataFrame({
        "Sentence": train_sentences,
        "NER Tags": train_pred_tags
    })
    train_df.to_csv("ner/results/train_predictions.csv", index=False)


    with torch.no_grad():
        for idx, (sentences, _) in enumerate(val_loader):
            if idx % 200 == 0:
                print(f"Procesando val batch {idx}")
            sentences = sentences.to(device)
            outputs = model(sentences)
            preds = torch.argmax(outputs, dim=-1)
            pred_tags = [idx_to_tag[idx.item()] for idx in preds.squeeze(0)]
            val_pred_tags.append(pred_tags)
            
            sentence_words = [idx_to_word.get(idx.item(), '<UNK>') for idx in sentences.squeeze(0).cpu()]
            sentence = " ".join(word for word in sentence_words if word != '<PAD>')
            val_sentences.append(sentence)

    val_df = pd.DataFrame({
        "Sentence": val_sentences,
        "NER Tags": val_pred_tags
    })
    val_df.to_csv("ner/results/val_predictions.csv", index=False)
    
    print("Predicciones guardadas en 'ner/results/train_predictions.csv' y 'ner/results/val_predictions.csv'")

if __name__ == "__main__":
    #main()
    #predict()
    get_results()
    