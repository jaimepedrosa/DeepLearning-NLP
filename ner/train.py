import torch


import torch

def train(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss,
        epochs,
        device):
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for sentences, tags in train_loader:
            sentences, tags = sentences.to(device), tags.to(device)
            
            optimizer.zero_grad()
            outputs = model(sentences)  # [batch_size, seq_len, tagset_size]
            loss_value = loss(outputs.view(-1, outputs.shape[-1]), tags.view(-1))
            loss_value.backward()
            optimizer.step()
            train_loss += loss_value.item()
            
            # Calcular accuracy
            preds = torch.argmax(outputs, dim=-1)  # [batch_size, seq_len]
            mask = tags != 0  # Ignorar padding (índice 0)
            train_correct += (preds[mask] == tags[mask]).sum().item()
            train_total += mask.sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        
        # Validación
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sentences, tags in val_loader:
                sentences, tags = sentences.to(device), tags.to(device)
                outputs = model(sentences)
                loss_value = loss(outputs.view(-1, outputs.shape[-1]), tags.view(-1))
                val_loss += loss_value.item()
                
                preds = torch.argmax(outputs, dim=-1)
                mask = tags != 0
                val_correct += (preds[mask] == tags[mask]).sum().item()
                val_total += mask.sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
