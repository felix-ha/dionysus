import time
from tqdm.autonotebook import tqdm
import os
from pathlib import Path
import datetime
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
)
from dataclasses import dataclass

from dionysus.utils import (
    compute_size,
    save_checkpoint,
    save_checkpoint_batch,
    time_pipeline,
)
from . import constants
from . import utils
from . import loss


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
    force_write_logs: bool = False
    tar_result: bool = False
    save_path: str = None
    add_time_to_save_path: bool = False
    model_name: str = None
    classification_metrics: dict = False
    class_names: list = None
    progress_bar: bool = True
    checkpoint_step: int = None  # save every checkpoint_step epoch
    checkpoint_step_batch: int = None  # save every checkpoint_step_batch batch
    checkpoint_path: str = None #  to continue training 
    batch_to_continue: int = None # batch to continue training 


    def __post_init__(self):
        if self.save_model:
            current_time = datetime.datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            if self.add_time_to_save_path:
                directory_name = f"{timestamp}_{self.model_name}"
            else:
                directory_name = f'{self.model_name}'
            # TODO fix naming or general handling of saving
            self.save_path_final = Path(self.save_path).joinpath(
                directory_name
            )
            self.save_path_final.mkdir(parents=True, exist_ok=True)
            logfile = os.path.join(self.save_path_final, constants.LOG_FILE)
        else:
            logfile = constants.LOG_FILE

        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            level=logging.INFO,
            handlers=[logging.FileHandler(logfile, mode="w")],
            force=self.force_write_logs,
        )

        if self.device == "gpu" or self.device == torch.device("cuda:0"):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"using device {self.device}")
        elif self.device == "cpu" or torch.device("cpu"):
            self.device = torch.device("cpu")
            logging.info(f"using device {self.device}")
        else:
            logging.info(f"device {self.device} is not available, using cpu instead")


        # TODO also enable to continue training for a given epoch. this is for huge datasets that are only trained on one epoch
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)

            if self.optimizer == "SGD":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            elif self.optimizer == "AdamW":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.batch_to_continue = checkpoint['batch']
            logging.info(f'continuing training at batch {self.batch_to_continue}')
        else:
            self.model.to(self.device)
            if self.optimizer == "SGD":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            elif self.optimizer == "AdamW":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)


    def loss(self, x, y):
        y_hat = self.model(x)
        return self.loss_func(y_hat, y), y_hat


@dataclass
class DistillConfig(TrainingConfig):
    # TODO: loss_func is needed for the super().__post_init__(), but is overwritten
    teacher: any = None
    alpha: float = None
    T: float = None

    def __post_init__(self):
        super().__post_init__()
        self.loss_func = loss.DistillationLoss(self.teacher, self.T, self.alpha)

    def loss(self, x, y):
        y_hat = self.model(x)
        return self.loss_func(y_hat, y, x), y_hat


def _setup_results(config):
    to_track = ["epoch", "epoch_time", "training_loss"]
    if config.validation_loader is not None:
        to_track.append("validation_loss")

    if config.classification_metrics:
        for name in constants.CLASSIFICATION_METRICS:
            to_track.append("training_" + name)
            if config.validation_loader is not None:
                to_track.append("validation_" + name)

    results = {}
    for item in to_track:
        results[item] = []

    return results


def train(config: TrainingConfig):
    results = _setup_results(config)
    time_training = 0
    logging.info("starting training")
    for epoch in tqdm(
        range(config.epochs), desc="epoch", disable=not config.progress_bar
    ):
        config.model = config.model.train()
        epoch_time, _ = run_epoch(config, results, epoch, prefix="training")
        time_training += epoch_time

        if config.validation_loader is not None:
            config.model = config.model.eval()
            with torch.no_grad():
                _, validation_result = run_epoch(
                    config, results, epoch, prefix="validation"
                )

        results["epoch"].append(epoch + 1)
        results["epoch_time"].append(epoch_time)

        if config.checkpoint_step is not None and epoch % config.checkpoint_step == 0:
            save_checkpoint("last", config, results, validation_result)

    if config.save_model:
        save_checkpoint("last", config, results, validation_result)

    logging.info(f"finished training, took {(time_training / 60 / 60):.3f} hours")

    size_mb = compute_size(config.model)
    logging.info(f"Model size (MB) - {size_mb:.4f}")

    time_avg_ms, time_std_ms = time_pipeline(config)
    logging.info(f"Average latency (ms) - {time_avg_ms:.2f} +/- {time_std_ms:.2f}")

    logging.info(f"results:\n {pd.DataFrame.from_dict(results)}")


def run_epoch(config: TrainingConfig, results: dict, epoch, prefix=""):
    if prefix == "training":
        data_loader = config.training_loader
    if prefix == "validation":
        data_loader = config.validation_loader
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    batch = 1
    for x, y in tqdm(
        data_loader, desc="batch", leave=False, disable=not config.progress_bar
    ):
        if config.batch_to_continue and batch < config.batch_to_continue:
            batch += 1
            continue
        
        x = utils.moveTo(x, config.device)
        y = utils.moveTo(y, config.device)

        loss, y_hat = config.loss(x, y)

        if config.model.training:
            loss.backward()
            config.optimizer.step()
            config.optimizer.zero_grad()

        running_loss.append(loss.item())

        if config.classification_metrics and isinstance(y, torch.Tensor):
            labels = y.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())

        if config.checkpoint_step_batch is not None and batch % config.checkpoint_step_batch == 0:
            save_checkpoint_batch(batch, config)

        batch += 1
        
    end = time.time()

    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    if config.classification_metrics:
        report_dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        macro_dict = report_dict["macro avg"]
        results[prefix + "_accuracy"].append(report_dict["accuracy"])
        results[prefix + "_macro_recall"].append(macro_dict["recall"])
        results[prefix + "_macro_precision"].append(macro_dict["precision"])
        results[prefix + "_macro_f1score"].append(macro_dict["f1-score"])

    results[prefix + "_loss"].append(np.mean(running_loss))
    time_elapsed = end - start
    if not config.progress_bar:
        if prefix == "training":
            logging.info(
                f"finished epoch {epoch+1}, took {(time_elapsed / 60 ):.3f} minutes"
            )

    if prefix == "validation":
        validation_result = (y_true, y_pred)
    else:
        validation_result = None

    return time_elapsed, validation_result
