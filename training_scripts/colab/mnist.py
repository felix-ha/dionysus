# !pip install git+https://github.com/felix-ha/dionysus-trainer.git@develop

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from dionysus.training import train, TrainingConfig
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
training_loader = DataLoader(training_dataset, batch_size=512, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=512)

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
)

train(train_config)
