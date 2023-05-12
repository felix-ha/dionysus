import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train(model, loss_func, optimizer, training_loader, validation_loader, epochs, device):
    to_track = ["epoch_time", "training_loss", "validation_loss"]
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


if __name__ == "__main__": 
    from sklearn.datasets import make_moons

    X_train, y_train = make_moons(n_samples=100, noise=0.1)
    X_validation, y_validation = make_moons(n_samples=100, noise=0.1)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
    validation_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32),
                                        torch.tensor(y_validation, dtype=torch.long))
    training_loader = DataLoader(train_dataset, shuffle=True)
    validation_loader = DataLoader(validation_dataset)

    in_features = 2
    out_features = 2
    model = nn.Linear(in_features, out_features)
    loss_func = nn.CrossEntropyLoss()

    device = torch.device("cpu")
    epochs = 10
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    results_pd = train(model, loss_func, optimizer, training_loader, validation_loader, epochs, device)

    print(results_pd)
