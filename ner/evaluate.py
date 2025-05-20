import torch

def evaluate(
        model, 
        sentence, 
        vocab, 
        tag_to_idx, 
        idx_to_tag, 
        device
        ):

    model.eval()
    
    words = sentence.split()
    sentence_idx = [vocab.get(word, vocab['<UNK>']) for word in words]
    input_tensor = torch.tensor(sentence_idx, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)  
        preds = torch.argmax(outputs, dim=-1) 
    
    pred_tags = [idx_to_tag[idx.item()] for idx in preds.squeeze(0)]
    
    result = dict(zip(words, pred_tags))
    
    return result