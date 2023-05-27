import time
from tqdm import tqdm
import os
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from models import moveTo

@dataclass
class TrainingConfig:
    model: any
    loss_func: any
    training_loader: DataLoader
    validation_loader: DataLoader = None
    lr: float = 0.001
    optimizer: str = "SGD"
    epochs: int = 2
    device: str = torch.device("cpu")
    save_model: bool = False
    save_path: str = None
    model_name: str = None
    score_funcs: dict = None

    def __post_init__(self):
        if self.optimizer == "SGD": 
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        if self.save_model:
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            Path(f"{self.save_path}_{timestamp}").mkdir(parents=True, exist_ok=False)
                    

def train(config: TrainingConfig):
    to_track = ["epoch_time", "training_loss"]
    if config.validation_loader is not None:
        to_track.append("validation_loss")

    if config.score_funcs is not None: 
        for name, _ in config.score_funcs.items():
            to_track.append("training_" + name)
            if config.validation_loader is not None:
                to_track.append("validation_" + name)
    
    results = {}
    for item in to_track:
        results[item] = []

    config.model.to(config.device)
    for epoch in tqdm(range(config.epochs), desc="Epoch"):
        config.model = config.model.train()
        epoch_time, x_sample = run_epoch(config, results, prefix="training")

        if config.validation_loader is not None:
            config.model = config.model.eval()
            with torch.no_grad():
                run_epoch(config, results, prefix="validation")
    
        results["epoch_time"].append(epoch_time)

    if config.save_model:
        # TODO fix error here_ RuntimeError: Parent directory C:\Users\FelixJobson\Desktop\priv\dl\runs\seq2seq does not exist.
        torch.save(config.model.state_dict(), os.path.join(config.save_path, config.model_name + ".pth"))
        try: 
            torch.onnx.export(config.model, x_sample, os.path.join(config.save_path, config.model_name +  ".onnx"), input_names=["features"], output_names=["logits"])
        except:
            print("saving onnx model failed")

    return pd.DataFrame.from_dict(results)  

def run_epoch(config: TrainingConfig, results: dict, prefix=""):
    if prefix == "training":
        data_loader = config.training_loader
    if prefix == "validation":
        data_loader = config.validation_loader
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for x, y in data_loader:         
        x = moveTo(x, config.device)
        y = moveTo(y, config.device)

        y_hat = config.model(x) 
        loss = config.loss_func(y_hat, y)

        if config.model.training:
            loss.backward()
            config.optimizer.step()
            config.optimizer.zero_grad()

        running_loss.append(loss.item())

        if config.score_funcs is not None and isinstance(y, torch.Tensor):
            #moving labels & predictions back to CPU for computing / storing predictions
            labels = y.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            #add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
        
    end =  time.time()

    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1: #We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    #Else, we assume we are working on a regression problem
    
    if config.score_funcs is not None:
        for name, score_func in config.score_funcs.items():
            try:
                results[prefix + "_" + name].append( score_func(y_true, y_pred) )
            except:
                results[prefix + "_" + name].append(float("NaN"))

    results[prefix + "_loss"].append(np.mean(running_loss))
    return end-start, x

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
