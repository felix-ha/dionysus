import os
import logging
import torch
import torch.nn as nn

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from dl.training import train, TrainingConfig, DistillationConfig
from dl.loss import DistilationLoss


def get_mnist_dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    trainset = datasets.MNIST(
        root=os.path.join(os.getcwd(), "data"),
        train=True,
        download=True,
        transform=transform,
    )
    testset = datasets.MNIST(
        root=os.path.join(os.getcwd(), "data"),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        trainset, batch_size=128, shuffle=True
    )
    test_loader = DataLoader(
        testset, batch_size=128, shuffle=False
    )

    return train_loader, test_loader


def train_teacher(n_features, n_classes, training_loader, validation_loader):
    teacher = nn.Sequential(
    nn.Flatten(),
    nn.Linear(n_features, 1200),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1200, 1200),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1200, n_classes),
)

    loss_func = nn.CrossEntropyLoss()

    train_config = TrainingConfig(
        model=teacher,
        epochs=100,
        loss_func=loss_func,
        training_loader=training_loader,
        validation_loader=validation_loader,
        save_model=True,
        save_path=os.path.join(os.getcwd(), "runs"),
        model_name="teacher_dropout",
        classification_metrics=True,
        class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        progress_bar=True,
        zip_result=False,
    )

    logging.info(f"start training of model: {train_config.model_name}")
    train(train_config)

    return train_config.model


n_features = 784
n_classes = 10

training_loader, validation_loader = get_mnist_dataloader()

# TODO: find teacher that learns mnist
teacher = train_teacher(n_features, n_classes, training_loader, validation_loader)

student = nn.Sequential(nn.Flatten(),nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, n_classes))

distilation_config = DistillationConfig(teacher=teacher, temperature=2, alpha=0.5)

train_config = TrainingConfig(
    model=student,
    epochs=5,
    loss_func=None,
    training_loader=training_loader,
    validation_loader=validation_loader,
    save_model=True,
    save_path=os.path.join(os.getcwd(), "runs"),
    model_name="student",
    classification_metrics=True,
    class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    progress_bar=True,
    zip_result=False,
    distillation_config=distilation_config
)

logging.info(f"start training of model: {train_config.model_name}")
train(train_config)
