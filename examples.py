import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training import train, cross_entropy_language_model
from data import LanguageModelDataset
from models import *


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
    corpus_file_training = 'data/small/training.txt'
    corpus_file_validation = 'data/small/validation.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    dataset_training = LanguageModelDataset(corpus_file_training, block_size=5)
    data_loader_training = DataLoader(dataset_training, batch_size=10, shuffle=True, generator=generator)
    dataset_validation= LanguageModelDataset(corpus_file_validation,
                                             block_size=5,
                                             vocabulary=dataset_training.vocabulary,
                                             encoder=dataset_training.encoder,
                                             decoder=dataset_training.decoder)
    data_loader_validation = DataLoader(dataset_validation, batch_size=10, shuffle=True, generator=generator)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BigramLanguageModel(vocab_size=len(dataset_training.vocabulary))
    m = model.to(device)

    loss_func = cross_entropy_language_model
    epochs = 2
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    result = train(model,
                    loss_func, 
                    optimizer,
                    training_loader=data_loader_training,
                    validation_loader=data_loader_validation,
                    epochs=epochs,
                    device=device)    
    print(result)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=10)[0].tolist())
    print(created_text)

def bigramV2():
    corpus_file_training = 'data/small/training.txt'
    corpus_file_validation = 'data/small/validation.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    dataset_training = LanguageModelDataset(corpus_file_training, block_size=5)
    data_loader_training = DataLoader(dataset_training, batch_size=10, shuffle=True, generator=generator)
    dataset_validation= LanguageModelDataset(corpus_file_validation,
                                             block_size=5,
                                             vocabulary=dataset_training.vocabulary,
                                             encoder=dataset_training.encoder,
                                             decoder=dataset_training.decoder)
    data_loader_validation = DataLoader(dataset_validation, batch_size=10, shuffle=True, generator=generator)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BigramLanguageModelV2(vocab_size=len(dataset_training.vocabulary), n_embd=5)
    m = model.to(device)

    loss_func = cross_entropy_language_model
    epochs = 2
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    result = train(model,
                    loss_func, 
                    optimizer,
                    training_loader=data_loader_training,
                    validation_loader=data_loader_validation,
                    epochs=epochs,
                    device=device)    
    print(result)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=10)[0].tolist())
    print(created_text)

def bigramV3():
    corpus_file_training = 'data/small/training.txt'
    corpus_file_validation = 'data/small/validation.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    n_embd = 5
    block_size = 5
    dataset_training = LanguageModelDataset(corpus_file_training, block_size=5)
    data_loader_training = DataLoader(dataset_training, batch_size=10, shuffle=True, generator=generator)
    dataset_validation= LanguageModelDataset(corpus_file_validation,
                                             block_size=block_size,
                                             vocabulary=dataset_training.vocabulary,
                                             encoder=dataset_training.encoder,
                                             decoder=dataset_training.decoder)
    data_loader_validation = DataLoader(dataset_validation, batch_size=10, shuffle=True, generator=generator)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BigramLanguageModelV3(vocab_size=len(dataset_training.vocabulary), n_embd=n_embd, block_size=block_size, device=device)
    m = model.to(device)

    loss_func = cross_entropy_language_model
    epochs = 2
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    result = train(model,
                    loss_func, 
                    optimizer,
                    training_loader=data_loader_training,
                    validation_loader=data_loader_validation,
                    epochs=epochs,
                    device=device)    
    print(result)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=block_size*2, block_size=block_size)[0].tolist())
    print(created_text)

def bigramV4():
    corpus_file_training = 'data/small/training.txt'
    corpus_file_validation = 'data/small/validation.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    n_embd = 5
    n_head = 3
    block_size = 5
    dataset_training = LanguageModelDataset(corpus_file_training, block_size=5)
    data_loader_training = DataLoader(dataset_training, batch_size=10, shuffle=True, generator=generator)
    dataset_validation= LanguageModelDataset(corpus_file_validation,
                                             block_size=block_size,
                                             vocabulary=dataset_training.vocabulary,
                                             encoder=dataset_training.encoder,
                                             decoder=dataset_training.decoder)
    data_loader_validation = DataLoader(dataset_validation, batch_size=10, shuffle=True, generator=generator)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BigramLanguageModelV4(vocab_size=len(dataset_training.vocabulary), n_embd=n_embd, head_size=n_head, block_size=block_size, device=device)
    m = model.to(device)

    loss_func = cross_entropy_language_model
    epochs = 2
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    result = train(model,
                    loss_func, 
                    optimizer,
                    training_loader=data_loader_training,
                    validation_loader=data_loader_validation,
                    epochs=epochs,
                    device=device)    
    print(result)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=block_size*2, block_size=block_size)[0].tolist())
    print(created_text)

def bigramV5():
    corpus_file_training = 'data/small/training.txt'
    corpus_file_validation = 'data/small/validation.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    n_embd = 8
    num_heads = 4
    n_head = 2
    block_size = 5
    dataset_training = LanguageModelDataset(corpus_file_training, block_size=5)
    data_loader_training = DataLoader(dataset_training, batch_size=10, shuffle=True, generator=generator)
    dataset_validation= LanguageModelDataset(corpus_file_validation,
                                             block_size=block_size,
                                             vocabulary=dataset_training.vocabulary,
                                             encoder=dataset_training.encoder,
                                             decoder=dataset_training.decoder)
    data_loader_validation = DataLoader(dataset_validation, batch_size=10, shuffle=True, generator=generator)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BigramLanguageModelV5(vocab_size=len(dataset_training.vocabulary), n_embd=n_embd, num_heads=num_heads,  head_size=n_head, block_size=block_size, device=device)
    m = model.to(device)

    loss_func = cross_entropy_language_model
    epochs = 2
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    result = train(model,
                    loss_func, 
                    optimizer,
                    training_loader=data_loader_training,
                    validation_loader=data_loader_validation,
                    epochs=epochs,
                    device=device)    
    print(result)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=block_size*2, block_size=block_size)[0].tolist())
    print(created_text)

def bigramV6():
    corpus_file_training = 'data/small/training.txt'
    corpus_file_validation = 'data/small/validation.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    n_embd = 8
    num_heads = 4
    block_size = 5
    dropout = 0.2
    dataset_training = LanguageModelDataset(corpus_file_training, block_size=5)
    data_loader_training = DataLoader(dataset_training, batch_size=10, shuffle=True, generator=generator)
    dataset_validation= LanguageModelDataset(corpus_file_validation,
                                             block_size=block_size,
                                             vocabulary=dataset_training.vocabulary,
                                             encoder=dataset_training.encoder,
                                             decoder=dataset_training.decoder)
    data_loader_validation = DataLoader(dataset_validation, batch_size=10, shuffle=True, generator=generator)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BigramLanguageModelV6(vocab_size=len(dataset_training.vocabulary), n_embd=n_embd, num_heads=num_heads, block_size=block_size, dropout=dropout, device=device)
    m = model.to(device)

    loss_func = cross_entropy_language_model
    epochs = 2
    lr = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    result = train(model,
                    loss_func, 
                    optimizer,
                    training_loader=data_loader_training,
                    validation_loader=data_loader_validation,
                    epochs=epochs,
                    device=device)    
    print(result)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=block_size*2, block_size=block_size)[0].tolist())
    print(created_text)

if __name__ == "__main__": 
    #feadforward_moon()
    #bigram()
    bigramV6()
