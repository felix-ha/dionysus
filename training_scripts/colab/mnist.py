# !pip install dionysus-trainer

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from dionysus.training import train, TrainingConfig


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


training_dataset, validation_dataset = get_mnist_datasets()
training_loader = DataLoader(training_dataset, batch_size=512, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=512)

model = Net()
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
    zip_result=True,
    save_path=save_path,
    model_name="mnist",
    progress_bar=False,
)

train(train_config)
