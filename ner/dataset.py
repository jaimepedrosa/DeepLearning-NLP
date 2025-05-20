import torch
from torch.utils.data import Dataset, DataLoader, random_split
from preprocessing import load_data, build_tag_map, build_vocab


class NERDataset(Dataset):
    def __init__(self, sentences, ner_tags, vocab, tag_to_idx):
        self.sentences = sentences
        self.ner_tags = ner_tags
        self.vocab = vocab
        self.tag_to_idx = tag_to_idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx].split()
        tags = self.ner_tags[idx]
        #assert len(sentence) == len(tags), f"Idx {idx}: {len(sentence)} palabras, {len(tags)} etiquetas, Sentence: {sentence}, Tags: {tags}"
        sentence_idx = [self.vocab.get(word, self.vocab['<UNK>']) for word in sentence]
        tags_idx = [self.tag_to_idx[tag] for tag in tags]
        
        return torch.tensor(sentence_idx, dtype=torch.long), torch.tensor(tags_idx, dtype=torch.long)
    

def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    tags = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-1)
    return sentences, tags


def get_dataloaders(file_path, batch_size=2, train_split=0.8):
    sentences, ner_tags = load_data(file_path)
    vocab = build_vocab(sentences)
    tag_to_idx, idx_to_tag = build_tag_map(ner_tags)

    dataset = NERDataset(sentences, ner_tags, vocab, tag_to_idx)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, vocab, tag_to_idx, idx_to_tag    