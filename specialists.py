import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from dl.training import TrainingConfig, train
from dl.data import *
from dl.models import *
import logging


def run_multiclass():
    from sklearn.datasets import make_classification

    n_features = 2
    n_classes = 4
    batch_size = 5

    X_train, y_train = make_classification(n_samples=100, 
                                           n_features = n_features, 
                                           n_redundant = 0,
                                           n_classes=n_classes, 
                                           n_clusters_per_class=1, 
                                           class_sep=1.3,
                                           random_state=6,
                                           weights=[1/n_classes for _ in range(n_classes)])
    
    df = pd.DataFrame(X_train)
    df.columns=['x1', 'x2']
    df['label'] = y_train 
    sns.scatterplot(x='x1', y='x2', hue='label', data=df)


    print(df)

    X_validation, y_validation = make_classification(n_samples=100, 
                                           n_features = n_features, 
                                           n_redundant = 0,
                                           n_classes=n_classes, 
                                           n_clusters_per_class=1, 
                                            weights=[1/n_classes for _ in range(n_classes)])
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
    validation_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32),
                                        torch.tensor(y_validation, dtype=torch.long))
    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)


    model =nn.Linear(n_features, n_classes) 
    loss_func = nn.CrossEntropyLoss()

    train_config = TrainingConfig(model=model,
                                  epochs=300,
                                   loss_func=loss_func, 
                                   training_loader=training_loader, 
                                   validation_loader=validation_loader,
                                   save_model=False,
                                   save_path=os.path.join(os.getcwd(), "runs"),
                                   model_name="multiclass", 
                                   classification_metrics = True,
                                   class_names = ['A', 'B', 'C', 'D'],
                                   progress_bar=True,
                                   zip_result=False)
    
    logging.info(f"start training of model: {train_config.model_name}")
    train(train_config)

    softmax = torch.ones([len(train_dataset), n_classes])
    with torch.no_grad():
        train_config.model.eval() 
        for i, (x, _) in enumerate(training_loader): 
            logits = train_config.model(x)
            sm = torch.softmax(logits, dim=-1)
            softmax[i*batch_size : i*batch_size+batch_size] = sm

    covariance = torch.cov(softmax.T)

    X = covariance.numpy()
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    print(kmeans.labels_)

    plt.show()


if __name__ == "__main__": 
    run_multiclass()