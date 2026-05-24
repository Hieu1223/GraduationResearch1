
import torch
import json
import os
from nltk.tokenize import WordPunctTokenizer
import numpy as np


def build_vocab_from_txt(file_path, embedding_dim):
    word2id = {}
    weights = []
    
    # 1. Add Special Tokens (Important for LSTMs)
    # <PAD> = 0 (all zeros), <UNK> = 1 (random numbers)
    word2id['<pad>'] = 0
    weights.append(np.zeros(embedding_dim))
    
    word2id['<unk>'] = 1
    weights.append(np.random.normal(scale=0.6, size=(embedding_dim,)))
    
    # 2. Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            
            if len(vector) != embedding_dim:
                continue # Skip header if present
                
            word2id[word] = len(weights)
            weights.append(vector)
            
    return word2id, torch.tensor(np.array(weights))


class Glove:
    def __init__(self, pretrained_file="glove.6B.300d.txt", embed_dim=300, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.matrix_path = os.path.join(cache_dir, f"matrix_{embed_dim}.pt")
        self.vocab_path = os.path.join(cache_dir, f"vocab_{embed_dim}.json")
        self.tk = WordPunctTokenizer()

        # Try to quick-load; otherwise, build and save
        if os.path.exists(self.matrix_path) and os.path.exists(self.vocab_path):
            self._load_checkpoint()
        else:
            print(f"No cache found. Processing {pretrained_file} (this may take a minute)...")
            # Assuming build_vocab_from_txt is defined elsewhere in your script
            self.word2id, self.weight_matrix = build_vocab_from_txt(pretrained_file, embed_dim)
            self._save_checkpoint()

        self.vocab_size = len(self.word2id)
        self.id2word = {int(v): k for k, v in self.word2id.items()}
        self.embed_dim = self.weight_matrix.shape[-1]



    def _save_checkpoint(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Save matrix as a Torch tensor for speed and precision
        if not isinstance(self.weight_matrix, torch.Tensor):
            self.weight_matrix = torch.tensor(self.weight_matrix)
        torch.save(self.weight_matrix, self.matrix_path)
        
        # Save dictionary as JSON
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.word2id, f)
        print(f"Saved quick-load cache to {self.cache_dir}")

    def _load_checkpoint(self):
        print("Quick-loading Glove from cache...")
        self.weight_matrix = torch.load(self.matrix_path)
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)

    def encode(self, text: str):
        text = text.lower()
        tokens = self.tk.tokenize(text)
        unk_idx = self.word2id.get('<unk>', 0) 
        return [int(self.word2id.get(token, unk_idx)) for token in tokens]

    def decode(self, indices: list[int]) -> str:
        tokens = [self.id2word.get(idx, '<unk>') for idx in indices]
        return " ".join(tokens)