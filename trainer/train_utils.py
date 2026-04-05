import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from .utils import tokenize_batch_and_pad


# ---------------------------------------------------------------------------
# TBPTT step
# ---------------------------------------------------------------------------

def train_tbptt_step(
    model, batch, chunk_size, embedding, optimizer, criterion, device
):
    """
    Truncated Backprop Through Time step.

    Inputs  = tokens at positions  [i  ..  i + chunk_size - 1]
    Targets = tokens at positions  [i+1 ..  i + chunk_size    ]

    The sequence must therefore be padded to length T+1 so the last
    input chunk always has a valid target slice.
    """
    data, mask, _ = batch                          # data: (B, T+1)

    data = data.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    B, T_plus1 = data.shape
    T          = T_plus1 - 1                       # usable input length

    all_embeddings = embedding(data[:, :-1]).to(torch.float32)  # (B, T, D)

    num_layers = model.lstm.num_layers
    hidden_dim = model.lstm.hidden_size

    states     = None
    batch_loss = 0.0
    step_count = 0

    optimizer.zero_grad()

    for i in range(0, T, chunk_size):
        inp_emb = all_embeddings[:, i : i + chunk_size]           # (B, L)
        tgt_tok = data[:, i + 1 : i + chunk_size + 1]             # (B, L)
        tgt_msk = mask[:, i + 1 : i + chunk_size + 1]             # (B, L)

        # Last chunk: input may be shorter than target slice — trim to match
        chunk_lengths  = tgt_msk.sum(dim=1).cpu()                # (B,)
        active_indices = (chunk_lengths > 0).nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            continue

        inp_emb        = inp_emb[active_indices]                  # (A, L, D)
        tgt_tok        = tgt_tok[active_indices]                  # (A, L)
        tgt_msk        = tgt_msk[active_indices]                  # (A, L)
        active_lengths = chunk_lengths[active_indices]

        # ---- carry hidden state for active sequences only ----------------
        if states is not None:
            h, c = states
            active_states = (h[:, active_indices, :], c[:, active_indices, :])
        else:
            active_states = None

        # ---- forward -----------------------------------------------------
        logits, _, new_states = model(inp_emb, active_lengths, active_states)


        L = logits.size(1)
        tgt_tok = tgt_tok[active_indices][:, :L]
        tgt_msk = tgt_msk[active_indices][:, :L]



        # ---- write updated states back into the full-batch buffer --------
        if states is None:
            h_full = torch.zeros(num_layers, B, hidden_dim, device=device)
            c_full = torch.zeros(num_layers, B, hidden_dim, device=device)
        else:
            h_full, c_full = states

        h_full = h_full.index_copy(1, active_indices.to(device), new_states[0])
        c_full = c_full.index_copy(1, active_indices.to(device), new_states[1])
        states = (h_full.detach(), c_full.detach())

        # ---- loss (only over non-pad positions) --------------------------
        logits_flat  = logits.reshape(-1, logits.size(-1))
        targets_flat = tgt_tok.reshape(-1)
        mask_flat    = tgt_msk.reshape(-1).bool()

        loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
        loss.backward()

        batch_loss += loss.item()
        step_count += 1

    if step_count > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return batch_loss, step_count


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    dataloader,
    chunk_size,
    embedding,
    optimizer,
    criterion,
    device,
    save_fn,
    save_every=200,
    log_every = 100
):
    model.train()
    total_loss, total_steps = 0.0, 0

    for batch_idx, batch in enumerate(dataloader):
        batch_loss, step_count = train_tbptt_step(
            model, batch, chunk_size, embedding, optimizer, criterion, device
        )
        total_loss  += batch_loss
        total_steps += step_count

        if (batch_idx + 1) % log_every == 0:
            avg = batch_loss / step_count if step_count else 0.0
            print(f"  batch {batch_idx + 1:>4d} | loss {avg:.4f} | steps {total_steps}")

        if (batch_idx + 1) % save_every == 0:
            save_fn(batch_idx + 1)

    return total_loss / total_steps if total_steps > 0 else 0.0

