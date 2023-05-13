import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training import train, cross_entropy_language_model
from data import LanguageModelDataset
from models import BigramLanguageModel


def feadforward_moon():
    from sklearn.datasets import make_moons

    X_train, y_train = make_moons(n_samples=10, noise=0.1)
    X_validation, y_validation = make_moons(n_samples=5, noise=0.1)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
    validation_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32),
                                        torch.tensor(y_validation, dtype=torch.long))
    training_loader = DataLoader(train_dataset, shuffle=True)
    validation_loader = DataLoader(validation_dataset)

    in_features = 2
    out_features = 2
    model = nn.Linear(in_features, out_features)
    loss_func = nn.CrossEntropyLoss()

    device = torch.device("cpu")
    epochs = 2
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    results_pd = train(model, loss_func, optimizer, training_loader, validation_loader, epochs, device)

    print(results_pd)

def bigram():
    corpus_file = 'data/small/training.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    dataset = LanguageModelDataset(corpus_file, block_size=5)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True, generator=generator)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BigramLanguageModel(vocab_size=len(dataset.vocabulary))
    m = model.to(device)

    loss_func = cross_entropy_language_model
    epochs = 2
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    result = train(model, loss_func, optimizer, training_loader=data_loader, validation_loader=None, epochs=epochs, device=device)
    print(result)

if __name__ == "__main__": 
    #feadforward_moon()
    bigram()
