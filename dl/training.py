"""Based on: Inside Deep Learning"""

import time
from tqdm.autonotebook import tqdm
import os
from pathlib import Path
import datetime
import zipfile
import logging
import tempfile
from time import perf_counter
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
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
    zip_result: bool = False
    save_path: str = None
    model_name: str = None
    classification_metrics: dict = False
    class_names: list = None
    progress_bar: bool = True
    checkpoint_epochs: list[int] = None

    def __post_init__(self):
        if self.optimizer == "SGD": 
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "AdamW": 
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if self.save_model:
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            # TODO fix naming or general handling of saving
            self.save_path_final = Path(self.save_path).joinpath(f"{timestamp}_{self.model_name}")
            self.save_path_final.mkdir(parents=True, exist_ok=False)
            logfile = os.path.join(self.save_path_final, constants.LOG_FILE)
        else:
            logfile = constants.LOG_FILE

        logging.basicConfig(
            format='%(asctime)s - %(message)s',
            level=logging.INFO,
            handlers=[logging.FileHandler(logfile, mode='w'),
            force=True]
            )

        if self.device == "gpu" or self.device == torch.device("cuda:0"):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"using device {self.device}")
        elif self.device == "cpu" or torch.device("cpu"):
            self.device = torch.device("cpu")
            logging.info(f"using device {self.device}")
        else:
            logging.info(f"device {self.device} is not available, using cpu instead")


def save_loss(results_pd, results_path):
    sns.lineplot(x="epoch", y="training_loss", data=results_pd, label="Training Loss")
    plot = sns.lineplot(x="epoch", y="validation_loss", data=results_pd, label="Validation Loss")
    # plt.title("Title")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
   # plt.show()
    fig = plot.get_figure()
    fig.savefig(results_path.joinpath("loss.png"))
    plt.close()

def save_metrics(results_pd, results_path, prefix=""):
    sns.set_style("darkgrid")
    sns.lineplot(x="epoch", y= prefix + "_accuracy", data=results_pd, label="Accuracy")
    sns.lineplot(x="epoch", y= prefix + "_macro_recall", data=results_pd, label="Macro Recall", linestyle='--')
    sns.lineplot(x="epoch", y= prefix + "_macro_precision", data=results_pd, label="Macro Precision", linestyle='--')
    plot =sns.lineplot(x="epoch", y= prefix + "_macro_f1score", data=results_pd, label="Macro F1-Score", color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
   # plt.show()
    fig = plot.get_figure()
    fig.savefig(results_path.joinpath(prefix + "_metrics.png"))
    plt.close()

def save_confusion_matrix(validation_result, labels, results_path):
    sns.set_style("white")
    y_true, y_pred = validation_result
    cm=confusion_matrix(y_true, y_pred)
    fig,ax=plt.subplots(figsize=(6,6))
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(cmap="Blues",values_format=".0f",ax=ax,colorbar=False)
    plt.xticks(rotation=90)
    plt.title("Confusion matrix")
    fig.savefig(results_path.joinpath("cm.png"))
    plt.close()


def compute_size(model):
    state_dict = model.state_dict()
    with tempfile.TemporaryDirectory() as tempdir:
        tmp_path = Path(tempdir).joinpath('model.pt')
        torch.save(state_dict, tmp_path)
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
    return size_mb

def time_pipeline(config, runs=100, warmup_runs=10):
    config.model.to(config.device)
    config.model = config.model.eval()
    with torch.no_grad():
        x, _ = next(iter(config.validation_loader))
        x = models.moveTo(x, config.device)
        latencies = []
        for _ in range(warmup_runs):
            _ = config.model(x)
        for _ in range(runs):
            start_time = perf_counter()
            _ = config.model(x)
            latency = perf_counter() - start_time
            latencies.append(latency)
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
    return time_avg_ms, time_std_ms


def save_checkpoint(epoch, config, results, validation_result, x_sample):
    subdirectory = "last" if epoch == "last" else f"epoch_{epoch}"
    save_path = Path(config.save_path_final).joinpath(subdirectory)
    save_path.mkdir(parents=True, exist_ok=False)

    results_pd = pd.DataFrame.from_dict(results)  

    # TODO move name of keys to constants
    torch.save({
    'epoch': epoch,
    'model_state_dict': config.model.state_dict(),
    'optimizer_state_dict': config.optimizer.state_dict(),
    'results' : results_pd,
    'validation_result': validation_result
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

    if config.classification_metrics: 
        for name in ['accuracy', 'macro_recall', 'macro_precision', 'macro_f1score']:
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
        epoch_time, _, x_sample = run_epoch(config, results, epoch, prefix="training")
        time_training += epoch_time

        if config.validation_loader is not None:
            config.model = config.model.eval()
            with torch.no_grad():
                _, validation_result, _ = run_epoch(config, results, epoch, prefix="validation")
    
        results["epoch"].append(epoch+1)
        results["epoch_time"].append(epoch_time)

        if config.checkpoint_epochs is not None and epoch in config.checkpoint_epochs:
            save_checkpoint(epoch, config, results, validation_result, x_sample)

    if config.save_model:
        save_checkpoint("last", config, results, validation_result, x_sample)

    if config.classification_metrics:
        y_true, y_pred = validation_result

        cm=confusion_matrix(y_true, y_pred)
        logging.info("confusion matrix: ")
        logging.info(f"\n{cm}")

        dummy_clf = DummyClassifier(strategy='most_frequent')
        dummy_clf.fit(np.zeros(len(y_true)), y_true)
        y_pred_dummy = dummy_clf.predict(np.zeros(len(y_true)))
        dummy_report = classification_report(y_true, y_pred_dummy, target_names=config.class_names, zero_division=0)
        logging.info("classification report baseline: ")
        logging.info(f"\n{dummy_report}")

        report = classification_report(y_true, y_pred, target_names=config.class_names, zero_division=0)
        logging.info("classification report: ")
        logging.info(f"\n{report}")
        
        if config.save_model:
            save_loss(results, config.save_path_final)
            save_metrics(results, config.save_path_final, "training")
            save_metrics(results, config.save_path_final, "validation")
            save_confusion_matrix(validation_result, labels=config.class_names, results_path=config.save_path_final)

    logging.info(f"finished training, took {(time_training / 60 / 60):.3f} hours")

    size_mb = compute_size(config.model)
    logging.info(f"Model size (MB) - {size_mb:.4f}")

    time_avg_ms, time_std_ms = time_pipeline(config)
    logging.info(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")

    if config.zip_result:
        zip_results(config)





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

        if config.classification_metrics and isinstance(y, torch.Tensor):
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
    
    if config.classification_metrics:
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        macro_dict = report_dict['macro avg']
        results[prefix + "_accuracy"].append(report_dict['accuracy'])
        results[prefix + "_macro_recall"].append(macro_dict['recall'])
        results[prefix + "_macro_precision"].append(macro_dict['precision'])
        results[prefix + "_macro_f1score"].append(macro_dict['f1-score'])


    results[prefix + "_loss"].append(np.mean(running_loss))
    time_elapsed = end-start
    if not config.progress_bar:
        if prefix == "training":
            logging.info(f"finished epoch {epoch-1}, took {(time_elapsed / 60 ):.3f} minutes")

    if config.epochs - 1 == epoch and prefix == "validation":
        logging.info(f"last {epoch}")
        validation_result = (y_true, y_pred)
    else:
        validation_result = None

    return time_elapsed, validation_result, x

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


# TODO Whole folder structure is saved atm, the results folder should be the only parent dir 

def zip_results(train_config):
    folder_name = Path(train_config.save_path_final).parts[-1]
    zip_file_name = Path(train_config.save_path_final).parent.joinpath(f'{folder_name}.zip')

    zip_file = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
    zipdir(train_config.save_path_final, zip_file)

def zipdir(path, ziph):
    for root, _, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))