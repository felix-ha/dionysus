import unittest
import os
import tempfile
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from src.dionysus.training import train, TrainingConfig, DistillConfig


class Test(unittest.TestCase):
    def test_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test Training
            n_features = 4
            n_classes = 3
            weights = [0.8, 0.15, 0.05]

            X, y = make_classification(
                n_samples=100,
                n_features=n_features,
                n_redundant=0,
                n_classes=n_classes,
                n_clusters_per_class=1,
                n_informative=3,
                class_sep=1,
                random_state=123,
                weights=weights,
            )
            X_train, X_validation, y_train, y_validation = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=123
            )

            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            )
            validation_dataset = TensorDataset(
                torch.tensor(X_validation, dtype=torch.float32),
                torch.tensor(y_validation, dtype=torch.long),
            )
            training_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            validation_loader = DataLoader(validation_dataset, batch_size=16)

            model = nn.Linear(n_features, n_classes)
            loss_func = nn.CrossEntropyLoss()

            save_path = os.path.join(temp_dir, "runs")

            train_config = TrainingConfig(
                model=model,
                epochs=3,
                loss_func=loss_func,
                training_loader=training_loader,
                validation_loader=validation_loader,
                save_model=True,
                classification_metrics=True,
                class_names=["A", "B", "C"],
                tar_result=True,
                save_path=save_path,
                model_name="ffw_moon",
                progress_bar=False,
                checkpoint_step=2,
            )

            train(train_config)

            assert os.path.exists(
                save_path
            ), f"save directory: {save_path} was no created"
            subdirs = [dirpath for dirpath, _, _ in os.walk(save_path)]
            assert subdirs[1].endswith(
                "ffw_moon"
            ), f"results directory: {subdirs[1]} was no created"
            tar_file = subdirs[1] + ".tar"
            assert os.path.isfile(tar_file), f"tar file: {tar_file} was not created"
            assert "info.log" in os.listdir(subdirs[1]), "logfile was not created"
            assert "cm.png" in os.listdir(
                subdirs[1]
            ), "classification matrix plot was not created"
            assert "loss.png" in os.listdir(subdirs[1]), "loss plot was not created"
            assert "training_metrics.png" in os.listdir(
                subdirs[1]
            ), "training metrics plot was not created"
            assert "validation_metrics.png" in os.listdir(
                subdirs[1]
            ), "validation metrics plot was not created"

            assert subdirs[2].endswith(
                "ffw_moon/last"
            ), f"results last directory: {subdirs[2]} was no created"
            assert "model.onnx" in os.listdir(subdirs[2]), "model.onnx was not created"
            assert "model.pt" in os.listdir(subdirs[2]), "model.pt was not created"

            # Test Inference
            model_inference = nn.Linear(n_features, n_classes)
            training_result_dict = torch.load(os.path.join(subdirs[2], "model.pt"))
            model_state_dict = training_result_dict["model_state_dict"]
            model.load_state_dict(model_state_dict)
            with torch.no_grad():
                model_inference.eval()
                _ = model_inference(torch.tensor(X_validation, dtype=torch.float32))

            # test distiller

            distilled_model = nn.Linear(n_features, n_classes)

            distill_config = DistillConfig(
                model=distilled_model,
                epochs=2,
                loss_func=loss_func,
                training_loader=training_loader,
                validation_loader=validation_loader,
                save_model=True,
                classification_metrics=True,
                class_names=["A", "B", "C"],
                tar_result=True,
                save_path=save_path,
                model_name="ffw_moon_distilled",
                progress_bar=False,
                teacher=model_inference,
                alpha=0.5,
                T=2.0,
            )

            train(distill_config)

            # TODO add asserts for distiller

            # TODO test continue training
