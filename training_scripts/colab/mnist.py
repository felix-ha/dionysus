#!pip install git+https://github.com/felix-ha/dionysus-trainer.git@develop

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from dionysus.training import train, TrainingConfig, DistillConfig
from dionysus.architectures import LeNet5


def get_mnist_datasets():
    try:
        training_dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        validation_dataset = torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
    except:
        training_dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        validation_dataset = torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
    return training_dataset, validation_dataset


training_dataset, validation_dataset = get_mnist_datasets()
training_loader = DataLoader(training_dataset, batch_size=512 * 4, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=512 * 4)

model = LeNet5()
loss_func = nn.CrossEntropyLoss()

save_path = "runs"

train_config = TrainingConfig(
    model=model,
    epochs=10,
    loss_func=loss_func,
    training_loader=training_loader,
    validation_loader=validation_loader,
    optimizer="AdamW",
    device="gpu",
    save_model=True,
    colab=True,
    classification_metrics=True,
    class_names=[str(i) for i in range(0, 10)],
    tar_result=True,
    save_path=save_path,
    model_name="LeNet-5",
    progress_bar=False,
    checkpoint_step=5,
)

train(train_config)

# set absolut path here
dict_path = "/content/runs/20231012_202618_LeNet-5/last"
teacher = LeNet5()
training_result_dict = torch.load(os.path.join(dict_path, "model.pt"))
model_state_dict = training_result_dict["model_state_dict"]
teacher.load_state_dict(model_state_dict)

model_distilled = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

distill_config = DistillConfig(
    model=model_distilled,
    epochs=10,
    loss_func=loss_func,
    training_loader=training_loader,
    validation_loader=validation_loader,
    optimizer="AdamW",
    device="gpu",
    save_model=True,
    colab=True,
    classification_metrics=True,
    class_names=[str(i) for i in range(0, 10)],
    tar_result=True,
    save_path=save_path,
    model_name="LeNet-5_distilled",
    progress_bar=False,
    checkpoint_step=2,
    teacher=teacher,
    alpha=0.5,
    T=2.0,
)

train(distill_config)
