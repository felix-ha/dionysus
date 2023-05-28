"""Based on: Inside Deep Learning"""

import time
from tqdm.autonotebook import tqdm
import os
from pathlib import Path
import datetime
import logging

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from . import constants, models


@dataclass
class TrainingConfig:
    model: any
    loss_func: any
    training_loader: DataLoader
    validation_loader: DataLoader = None
    lr: float = 0.001
    optimizer: str = "SGD"
    epochs: int = 2
    device: str = "cpu"
    save_model: bool = False
    save_path: str = None
    model_name: str = None
    score_funcs: dict = None
    progress_bar: bool = True
    checkpoint_epochs: list[int] = None

    def __post_init__(self):
        if self.optimizer == "SGD": 
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        if self.save_model:
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            # TODO fix naming or general handling of saving
            self.save_path_final = Path(self.save_path).joinpath(f"{timestamp}_{self.model_name}")
            self.save_path_final.mkdir(parents=True, exist_ok=False)
            handlers = [logging.FileHandler(os.path.join(self.save_path_final, constants.LOG_FILE), mode='w'), logging.StreamHandler()]
        else: 
            handlers =  [logging.StreamHandler()]

        logging.basicConfig(
            format='%(asctime)s - %(message)s',
            level=logging.INFO,
            handlers=handlers
            )

        if self.device == "gpu" or self.device == torch.device("cuda:0"):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"using device {self.device}")
        elif self.device == "cpu" or torch.device("cpu"):
            self.device = torch.device("cpu")
            logging.info(f"using device {self.device}")
        else:
            logging.info(f"device {self.device} is not available, using cpu instead")


def save_checkpoint(epoch, config, results, x_sample):
    subdirectory = "last" if epoch == "last" else f"epoch_{epoch}"
    save_path = Path(config.save_path_final).joinpath(subdirectory)
    save_path.mkdir(parents=True, exist_ok=False)

    results_pd = pd.DataFrame.from_dict(results)  

    # TODO move name of keys to constants
    torch.save({
    'epoch': epoch,
    'model_state_dict': config.model.state_dict(),
    'optimizer_state_dict': config.optimizer.state_dict(),
    'results' : results_pd 
    }, os.path.join(save_path, constants.CHECKPOINT_FILE))
    logging.info("saved result dict")

    try: 
        torch.onnx.export(config.model, x_sample, os.path.join(save_path, "model.onnx"), input_names=["features"], output_names=["logits"])
        logging.info("saved onnx model")
    except:
        logging.warn("saving onnx model failed")      

    if epoch == "last":
        logging.info(f"results:\n{results_pd}")          


def train(config: TrainingConfig):
    logging.info("starting training")
    to_track = ["epoch", "epoch_time", "training_loss"]
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

    time_training = 0
    config.model.to(config.device)
    for epoch in tqdm(range(config.epochs), desc="epoch", disable = not config.progress_bar):
        config.model = config.model.train()
        epoch_time, x_sample = run_epoch(config, results, epoch, prefix="training")
        time_training += epoch_time

        if config.validation_loader is not None:
            config.model = config.model.eval()
            with torch.no_grad():
                run_epoch(config, results, epoch, prefix="validation")
    
        results["epoch"].append(epoch+1)
        results["epoch_time"].append(epoch_time)

        if config.checkpoint_epochs is not None and epoch in config.checkpoint_epochs:
            save_checkpoint(epoch, config, results, x_sample)

    logging.info(f"finished training, took {(time_training / 60 / 60):.3f} hours")

    if config.save_model:
        save_checkpoint("last", config, results, x_sample)


def run_epoch(config: TrainingConfig, results: dict, epoch, prefix=""):
    # TODO move strings to config
    if prefix == "training":
        data_loader = config.training_loader
    if prefix == "validation":
        data_loader = config.validation_loader
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for x, y in tqdm(data_loader, desc="batch", leave=False, disable = not config.progress_bar):      
        x = models.moveTo(x, config.device)
        y = models.moveTo(y, config.device)

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
    time_elapsed = end-start
    if not config.progress_bar:
        if prefix == "training":
            logging.info(f"finished epoch {epoch}, took {(time_elapsed / 60 ):.3f} minutes")
    return time_elapsed, x

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
