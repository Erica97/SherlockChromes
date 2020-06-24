import sys
import torch.optim as optim

from train.train import train

def main(
        data,
        model,
        loss,
        sampling_fn,
        collate_fn,
        optimizer_kwargs,
        lr_scheduler_kwargs,
        train_kwargs,
        device):
    optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)

    train(
        data,
        model,
        optimizer,
        loss,
        sampling_fn,
        collate_fn,
        device,
        **train_kwargs)
