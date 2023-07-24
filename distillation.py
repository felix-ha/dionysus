import os
import logging
import torch
import torch.nn as nn

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from dl.training import train, TrainingConfig, DistillationConfig
from dl.loss import DistilationLoss


def get_dataloader(n_features, n_classes):
    weights = [0.2, 0.3, 0.5]

    X, y = make_classification(
        n_samples=300,
        n_features=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_informative=2,
        class_sep=2,
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
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

    return training_loader, validation_loader


def train_teacher(n_features, n_classes, training_loader, validation_loader):
    teacher = nn.Sequential(
        nn.Linear(n_features, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, n_classes),
    )

    loss_func = nn.CrossEntropyLoss()

    train_config = TrainingConfig(
        model=teacher,
        epochs=10,
        loss_func=loss_func,
        training_loader=training_loader,
        validation_loader=validation_loader,
        save_model=True,
        save_path=os.path.join(os.getcwd(), "runs"),
        model_name="teacher",
        classification_metrics=True,
        class_names=["A", "B", "C"],
        progress_bar=True,
        zip_result=False,
    )

    logging.info(f"start training of model: {train_config.model_name}")
    train(train_config)

    return train_config.model


n_features = 2
n_classes = 3

training_loader, validation_loader = get_dataloader(n_features, n_classes)
teacher = train_teacher(n_features, n_classes, training_loader, validation_loader)

student = nn.Sequential(nn.Linear(n_features, 8), nn.ReLU(), nn.Linear(8, n_classes))

distilation_config = DistillationConfig(teacher=teacher, temperature=2, alpha=0.5)

train_config = TrainingConfig(
    model=student,
    epochs=2,
    loss_func=None,
    training_loader=training_loader,
    validation_loader=validation_loader,
    save_model=True,
    save_path=os.path.join(os.getcwd(), "runs"),
    model_name="student",
    classification_metrics=True,
    class_names=["A", "B", "C"],
    progress_bar=True,
    zip_result=False,
    distillation_config=distilation_config
)

logging.info(f"start training of model: {train_config.model_name}")
train(train_config)
