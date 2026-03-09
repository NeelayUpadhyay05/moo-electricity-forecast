import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
        output_dim=1
    ):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """

        _, (hidden, _) = self.lstm(x)

        # Take last hidden state from final layer
        last_hidden = hidden[-1]  # shape: (batch_size, hidden_dim)

        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)  # shape: (batch_size, output_dim)

        # Squeeze the last dimension for single-step forecasting so the output
        # shape (batch,) matches the scalar target produced by LoadDataset.
        if out.size(-1) == 1:
            out = out.squeeze(-1)

        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())