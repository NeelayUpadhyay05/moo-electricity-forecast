import torch


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, save_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss, model):

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True