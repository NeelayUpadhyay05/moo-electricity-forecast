import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, seq_len=24, conv_channels=16, lstm_hidden=64, lstm_layers=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.conv(x)    # (batch, conv_channels, seq_len)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, conv_channels)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(1)
