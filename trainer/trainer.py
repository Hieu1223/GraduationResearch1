import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from .utils import tokenize_batch_and_pad
from .train_utils import train_epoch

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        embed_model,
        optimizer,
        criterion,
        dataset,
        save_dir="checkpoints",
        max_checkpoints=3,
    ):
        self._raw_model     = model
        self._raw_embedding = embed_model

        self.model     = torch.compile(model)
        self.embedding = torch.compile(embed_model)

        self.loss_fn   = criterion
        self.optimizer = optimizer
        self.dataset   = dataset
        self.tokenizer = tokenizer

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.max_checkpoints: int       = max_checkpoints
        self.checkpoint_history: list[str] = []

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def _collate(self, batch):
        return tokenize_batch_and_pad(batch, self.tokenizer)

    def _make_dataloader(
        self,
        batch_size: int,
        device: str,
        skip_batches: int = 0,
    ) -> DataLoader:
        num_workers = min(4, os.cpu_count() or 1)


        indices   = torch.randperm(len(self.dataset)).tolist()
        remaining = indices[skip_batches * batch_size:]

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=remaining,
            pin_memory=(device == "cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            collate_fn=self._collate,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, batch_idx: int, prefix: str = "ckpt"):
        filename  = f"{prefix}_epoch_{epoch}_batch_{batch_idx}.pth"
        full_path = os.path.join(self.save_dir, filename)

        torch.save(
            {
                "epoch":                epoch,
                "batch_idx":            batch_idx,
                "model_state_dict":     self._raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            full_path,
        )

        self.checkpoint_history.append(full_path)
        print(f"Saved: {full_path}")

        while len(self.checkpoint_history) > self.max_checkpoints:
            oldest = self.checkpoint_history.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)
                print(f"Removed: {oldest}")

    def load_checkpoint(self, filename: str, device: str = "cuda"):
        """Returns (epoch, batch_idx, rng_state) or (0, 0, None) on failure."""
        full_path = os.path.join(self.save_dir, filename)

        if not os.path.exists(full_path):
            print(f"Checkpoint not found: {full_path}")
            return 0, 0, None

        ckpt = torch.load(full_path, map_location=device)
        self._raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        print(f"Resumed from {full_path} (epoch {ckpt['epoch']}, batch {ckpt['batch_idx']})")
        return ckpt["epoch"], ckpt["batch_idx"]

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int     = 10,
        chunk_size: int = 50,
        batch_size: int = 16,
        device: str     = "cuda",
        resume_file     = None,
    ):
        self.model.to(device)
        self.embedding.to(device)

        start_epoch, start_batch, rng_state = 0, 0, None
        if resume_file:
            start_epoch, start_batch = self.load_checkpoint(
                resume_file, device
            )

        for epoch in range(start_epoch, epochs):
            print(f"\n--- Epoch {epoch}/{epochs} ---")

            skip       = start_batch if epoch == start_epoch else 0
            dataloader = self._make_dataloader(batch_size, device, skip)

            avg_loss = train_epoch(
                self.model,
                dataloader,
                chunk_size,
                self.embedding,
                self.optimizer,
                self.loss_fn,
                device,
                save_fn=lambda bidx: self.save_checkpoint(epoch, skip + bidx),
            )

            print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")
            self.save_checkpoint(epoch + 1, batch_idx=0)


class ClassifierTrainer:
    def __init__(self, model, classifier, glove, embedding, device, trait_idx=0, lr=1e-4):
        self.model = model
        self.classifier = classifier
        self.glove = glove
        self.embedding = embedding
        self.device = device
        self.trait_idx = trait_idx
        
        self.optimizer = torch.optim.Adam(
            list(self.classifier.parameters()) + list(self.model.parameters()), 
            lr=lr
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, dataset, sample_range=(0, 1600)):
        self.model.train()
        self.classifier.train()
        total_loss = 0
        
        
        # Slicing the dataset based on provided range
        data_subset = list(zip(*dataset[sample_range[0] : sample_range[1]]))
        
        for text, traits in data_subset:
            self.optimizer.zero_grad()
            
            # Target for specific trait
            target_val = int(traits[self.trait_idx])
            target = torch.tensor([target_val], dtype=torch.long).to(self.device)
            
            # Preprocessing
            tokens = self.glove.encode(text)
            tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
            emb = self.embedding(tokens).to(torch.float32)
            seq_lens = torch.tensor([tokens.size(1)], device='cpu')
            
            # Forward
            _, (h, c) = self.model(emb, seq_lens, fc_out=False)
            logits = self.classifier(h[-1]) 
            
            loss = self.criterion(logits, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(data_subset)

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")



