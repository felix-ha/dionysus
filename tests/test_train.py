import unittest
import os 
import tempfile
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
import torch 
import torch.nn as nn

from src.dionysus.training import train, TrainingConfig


class Test(unittest.TestCase):      
    def test_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            X_train, y_train = make_moons(n_samples=10, noise=0.1)
            X_validation, y_validation = make_moons(n_samples=5, noise=0.1)
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.long))
            validation_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32),
                                                torch.tensor(y_validation, dtype=torch.long))
            training_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            validation_loader = DataLoader(validation_dataset, batch_size=32)
            in_features = 2
            out_features = 2
            model = nn.Linear(in_features, out_features)
            loss_func = nn.CrossEntropyLoss()

            save_path = os.path.join(temp_dir, "runs")

            train_config = TrainingConfig(model=model,
                                        epochs=2,
                                        loss_func=loss_func, 
                                        training_loader=training_loader, 
                                        validation_loader=validation_loader,
                                        save_model=True,
                                        zip_result=True,
                                        save_path=save_path,
                                        model_name="ffw_moon",
                                        progress_bar=False)

            train(train_config)

            assert os.path.exists(save_path), f"save directory: {save_path} was no created"
            subdirs = [dirpath for dirpath, _, _ in os.walk(save_path)]
            assert subdirs[1].endswith("ffw_moon"), f"results directory: {subdirs[0]} was no created"
            assert subdirs[2].endswith("ffw_moon/last"), f"results last directory: {subdirs[1]} was no created"
            assert "info.log" in os.listdir(subdirs[1]), "logfile was not created"
            assert "model.onnx" in os.listdir(subdirs[2]), "model.onnx was not created"
            assert "model.pt" in os.listdir(subdirs[2]), "model.pt was not created"
