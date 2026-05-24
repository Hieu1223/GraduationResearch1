import torch
from torch.nn.utils.rnn import pad_sequence


def tokenize_batch_and_pad(batch, tk):
    tokens  = [torch.tensor(tk.encode(sentence), dtype=torch.long) for sentence in batch]
    lengths = torch.tensor([len(t) for t in tokens])

    padded_batch = pad_sequence(tokens, batch_first=True, padding_value=0)

    B, T = padded_batch.shape
    padded_batch = torch.cat(
        [padded_batch, torch.zeros(B, 1, dtype=torch.long)], dim=1
    )  # (B, T+1)

    mask = torch.arange(T + 1).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T+1)

    return padded_batch, mask, lengths


def load_model_from_trainer_checkpoint(model, file_name : str):
    with open(file_name,'rb') as f:
        model.load_state_dict(torch.load(f)["model_state_dict"])