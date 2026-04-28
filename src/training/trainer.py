import torch
from tqdm import tqdm
from torch import amp


def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    scaler=None, max_grad_norm=1.0):
    model.train()

    total_loss = 0.0
    n_processed = 0
    use_amp = (scaler is not None)

    for x, y in tqdm(dataloader, desc="  train", leave=False,
                     unit="batch", ncols=90):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with amp.autocast(device_type="cuda"):
                outputs = model(x)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_processed += x.size(0)

    return total_loss / n_processed


def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    n_processed = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="    val", leave=False,
                         unit="batch", ncols=90):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            n_processed += x.size(0)

    return total_loss / n_processed
