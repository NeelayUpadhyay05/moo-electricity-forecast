import numpy as np
import torch
from torch.utils.data import Dataset


class GlobalLoadDataset(Dataset):
    def __init__(self, dataframe, input_window=24, output_window=24):
        """
        Global multi-series dataset.
        Each sample corresponds to one household window.
        """

        self.data = dataframe.values  # shape: (time_steps, n_households)
        self.input_window = input_window
        self.output_window = output_window

        self.time_steps = self.data.shape[0]
        self.n_households = self.data.shape[1]

        self.window_size = self.input_window + self.output_window

        # number of valid windows per household
        self.samples_per_household = self.time_steps - self.window_size + 1

        # total samples across all households
        self.total_samples = self.samples_per_household * self.n_households

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):

        household_index = idx // self.samples_per_household
        local_index = idx % self.samples_per_household

        start = local_index
        end_input = start + self.input_window
        end_output = end_input + self.output_window

        series = self.data[:, household_index]

        x = series[start:end_input]
        y = series[end_input:end_output]

        # reshape for LSTM: (seq_len, features=1)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y