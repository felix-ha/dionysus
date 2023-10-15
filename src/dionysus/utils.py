import tempfile
from time import perf_counter
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from dionysus import constants


import pandas as pd
import torch


import logging
import os
from pathlib import Path
import tarfile


def tar_folder(config):
    """
    Creates a .tar file for a given folder
    """
    path = config.save_path_final
    with tarfile.open(f"{path}.tar", "w") as tar:
        tar.add(path, arcname=os.path.basename(path))


def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj


def save_checkpoint(epoch, config, results, validation_result, x_sample):
    save_path = Path(config.save_path_final).joinpath("last")
    save_path.mkdir(parents=True, exist_ok=True)

    results_pd = pd.DataFrame.from_dict(results)

    # TODO move name of keys to constants
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": config.model.state_dict(),
            "optimizer_state_dict": config.optimizer.state_dict(),
            "results": results_pd,
            "validation_result": validation_result,
        },
        os.path.join(save_path, constants.CHECKPOINT_FILE),
    )
    logging.info("saved result dict")

    try:
        config.model.eval()
        torch.onnx.export(
            config.model,
            x_sample,
            os.path.join(save_path, "model.onnx"),
            input_names=["features"],
            output_names=["logits"],
        )
        logging.info("saved onnx model")
    except:
        logging.warn("saving onnx model failed")

    save_loss(results, config.save_path_final)
    
    if config.classification_metrics:
        create_classifcation_report(results, validation_result, config)

    if config.tar_result:
        tar_folder(config)


def time_pipeline(config, runs=100, warmup_runs=10):
    config.model.to(config.device)
    config.model = config.model.eval()
    with torch.no_grad():
        x, _ = next(iter(config.validation_loader))
        x = moveTo(x, config.device)
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


def compute_size(model):
    state_dict = model.state_dict()
    with tempfile.TemporaryDirectory() as tempdir:
        tmp_path = Path(tempdir).joinpath("model.pt")
        torch.save(state_dict, tmp_path)
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
    return size_mb


def save_confusion_matrix(validation_result, labels, results_path):
    sns.set_style("white")
    y_true, y_pred = validation_result
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".0f", ax=ax, colorbar=False)
    plt.xticks(rotation=90)
    plt.title("Confusion matrix")
    fig.savefig(results_path.joinpath("cm.png"))
    plt.close()


def save_metrics(results_pd, results_path, prefix=""):
    sns.set_style("darkgrid")
    sns.lineplot(x="epoch", y=prefix + "_accuracy", data=results_pd, label="Accuracy")
    sns.lineplot(
        x="epoch",
        y=prefix + "_macro_recall",
        data=results_pd,
        label="Macro Recall",
        linestyle="--",
    )
    sns.lineplot(
        x="epoch",
        y=prefix + "_macro_precision",
        data=results_pd,
        label="Macro Precision",
        linestyle="--",
    )
    plot = sns.lineplot(
        x="epoch",
        y=prefix + "_macro_f1score",
        data=results_pd,
        label="Macro F1-Score",
        color="r",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    fig = plot.get_figure()
    fig.savefig(results_path.joinpath(prefix + "_metrics.png"))
    plt.close()


def save_loss(results_pd, results_path):
    sns.lineplot(x="epoch", y="training_loss", data=results_pd, label="Training Loss")
    plot = sns.lineplot(
        x="epoch", y="validation_loss", data=results_pd, label="Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig = plot.get_figure()
    fig.savefig(results_path.joinpath("loss.png"))
    plt.close()


def create_classifcation_report(results, validation_result, config):
    y_true, y_pred = validation_result

    cm = confusion_matrix(y_true, y_pred)
    logging.info("confusion matrix: ")
    logging.info(f"\n{cm}")

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(np.zeros(len(y_true)), y_true)
    y_pred_dummy = dummy_clf.predict(np.zeros(len(y_true)))
    dummy_report = classification_report(
        y_true, y_pred_dummy, target_names=config.class_names, zero_division=0
    )
    logging.info("classification report baseline: ")
    logging.info(f"\n{dummy_report}")

    report = classification_report(
        y_true, y_pred, target_names=config.class_names, zero_division=0
    )
    logging.info("classification report: ")
    logging.info(f"\n{report}")

    if config.save_model:
        save_metrics(results, config.save_path_final, "training")
        save_metrics(results, config.save_path_final, "validation")
        save_confusion_matrix(
            validation_result,
            labels=config.class_names,
            results_path=config.save_path_final,
        )
