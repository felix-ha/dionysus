
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import optuna
from dl.training import TrainingConfig, train


def objective(trial, X_train, n_classes):
    X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
    validation_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32),
                                        torch.tensor(y_validation, dtype=torch.long))
    training_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=256)

    n_hidden = trial.suggest_int('hidden_dim', 2, 10)
    p_dropout = trial.suggest_float('p_dropout', 0, 1)

    model = nn.Sequential(nn.Linear(n_features, n_hidden), 
                        nn.Tanh(), nn.Dropout(p_dropout),
                        nn.Linear(n_hidden, n_classes)) 

    train_config = TrainingConfig(model=model,
                                    epochs=25,
                                    loss_func=nn.CrossEntropyLoss(), 
                                    training_loader=training_loader, 
                                    validation_loader=validation_loader, 
                                    classification_metrics = True,
                                    class_names = ['A', 'B', 'C'],
                                    progress_bar=False)

    results = train(train_config)
    metric = results['validation_accuracy'][-1]
    return metric


n_features = 2
n_classes = 3
weights = [0.7, 0.15, 0.15]

X, y = make_classification(n_samples=200, 
                            n_features = n_features, 
                            n_redundant = 0,
                            n_classes=n_classes, 
                            n_clusters_per_class=1,
                            n_informative=2, 
                            class_sep = 1.2,
                            random_state=123,
                            weights=weights)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                    torch.tensor(y_test, dtype=torch.long))
training_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)


study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X_train, n_classes), n_trials=5)
print(study.best_params)
