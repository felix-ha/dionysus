import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model: any
    loss_func: any
    training_loader: DataLoader
    validation_loader: DataLoader
    lr: float = 0.001
    optimizer: str = "SGD"
    epochs: int = 2
    device: str = torch.device("cpu")
    save_model: bool = False
    save_path: str = None

    def __post_init__(self):
        if self.optimizer == "SGD": 
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        


def train(config: TrainingConfig):
    to_track = ["epoch_time", "training_loss"]
    if config.validation_loader is not None:
        to_track.append("validation_loss")
    results = {}
    for item in to_track:
        results[item] = []

    config.model.to(config.device)
    for epoch in tqdm(range(config.epochs), desc="Epoch"):
        config.model = config.model.train()
        epoch_time = run_epoch(config, results, prefix="training")

        if config.validation_loader is not None:
            config.model = config.model.eval()
            with torch.no_grad():
                run_epoch(config, results, prefix="validation")
    
        results["epoch_time"].append(epoch_time)

    if config.save_model:
        torch.save(config.model.state_dict(), config.save_path)

    return pd.DataFrame.from_dict(results)  

def run_epoch(config: TrainingConfig, results: dict, prefix=""):
    if prefix == "training":
        data_loader = config.training_loader
    if prefix == "validation":
        data_loader = config.validation_loader
    running_loss = []
    start = time.time()
    for x, y in data_loader:         
        x = x.to(config.device)
        y = y.to(config.device)

        y_hat = config.model(x) 
        loss = config.loss_func(y_hat, y)

        if config.model.training:
            loss.backward()
            config.optimizer.step()
            config.optimizer.zero_grad()

        running_loss.append(loss.item())
    end =  time.time()
    results[prefix + "_loss"].append(np.mean(running_loss))
    return end-start

def cross_entropy_language_model(logits, targets):
    """
    Removes the time dimension for logits and targets and computes the cross entropy loss
    For the F.cross_entropy function, the inputs are predicted unnormalized logits and output are ground truth class indices or class probabilities
    """
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return loss
