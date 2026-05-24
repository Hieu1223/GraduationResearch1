import torch
from torch import nn


class SentimentAnalysisLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=1024, num_layers=3, dropout=0.3,dtype = torch.float32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            dtype=dtype
        )

        self.fc = nn.Linear(hidden_dim, vocab_size,dtype=dtype)

    def forward(self, x,lengths, h=None,fc_out = True):
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, h = self.lstm(packed_x, h)      
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        if fc_out:
            logits = self.fc(out) 
            return logits,out, h
        else:
            return out, h


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_labels,hidden_dim =(64,),dropout =0.3):
        super().__init__()
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim[0]),)
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))
        for i in range(len(hidden_dim)-1):
            modules.append(nn.Linear(hidden_dim[i],hidden_dim[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_dim[-1], num_labels))
        self.stack = nn.Sequential(*modules)

    def forward(self, x):
        return self.stack(x)  # logits