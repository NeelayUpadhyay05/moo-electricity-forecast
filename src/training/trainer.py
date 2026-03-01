import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validation", leave=False):
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)

def compute_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))