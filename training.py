import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def train(model, loss_func, optimizer, training_loader, validation_loader, epochs, device):
    to_track = ["epoch_time", "training_loss"]
    if validation_loader is not None:
        to_track.append("validation_loss")
    results = {}
    for item in to_track:
        results[item] = []

    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model = model.train()
        epoch_time = run_epoch(model, loss_func, optimizer, training_loader, device, results, prefix="training")

        if validation_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, loss_func, optimizer, validation_loader, device, results, prefix="validation")
    
        results["epoch_time"].append(epoch_time)

    return pd.DataFrame.from_dict(results)  

def run_epoch(model, loss_func, optimizer, data_loader, device, results, prefix=""):
    running_loss = []
    start = time.time()
    for x, y in data_loader:         
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x) 
        loss = loss_func(y_hat, y)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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
