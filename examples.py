import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

from training import TrainingConfig, train, cross_entropy_language_model
from data import LanguageModelDataset, LanguageNameDataset, LargestDigit, LargestDigitVariable, pad_and_pack
from models import *

import os


def feadforward_moon():
    from sklearn.datasets import make_moons

    X_train, y_train = make_moons(n_samples=10, noise=0.1)
    X_validation, y_validation = make_moons(n_samples=5, noise=0.1)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
    validation_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32),
                                        torch.tensor(y_validation, dtype=torch.long))
    training_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=2,)

    in_features = 2
    out_features = 2
    model = nn.Linear(in_features, out_features)
    loss_func = nn.CrossEntropyLoss()

    train_config = TrainingConfig(model=model,
                                   loss_func=loss_func, 
                                   training_loader=training_loader, 
                                   validation_loader=validation_loader,
                                   save_model=True,
                                   save_path=os.path.join(os.getcwd(), "runs", "trainings_run"),
                                   model_name="test", 
                                   score_funcs= {'accuracy': accuracy_score})
    results_pd = train(train_config)

    print(results_pd)

def run_RNN():
    dataset = LanguageNameDataset()

    train_data, test_data = torch.utils.data.random_split(dataset, (len(dataset)-300, 300))
    data_loader_training = DataLoader(train_data, batch_size=1, shuffle=True)
    data_loader_validation = DataLoader(test_data, batch_size=1, shuffle=False)

    dim_embeddings = 2 #64
    vocab_size = dataset.vocab_size
    hidden_nodes = 2 #256
    n_classes = len(dataset.label_names)

    model = RNN(vocab_size, dim_embeddings, hidden_nodes, n_classes)

    loss_func = nn.CrossEntropyLoss()

    train_config = TrainingConfig(model=model, loss_func=loss_func, training_loader=data_loader_training, validation_loader=data_loader_validation)
    results_pd = train(train_config)  
    print(results_pd)

def run_RNN_packed():
    dataset = LanguageNameDataset()

    train_data, test_data = torch.utils.data.random_split(dataset, (len(dataset)-300, 300))
    data_loader_training = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=pad_and_pack)
    data_loader_validation = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=pad_and_pack)

    dim_embeddings = 2 #64
    vocab_size = dataset.vocab_size
    hidden_nodes = 2 #256
    n_classes = len(dataset.label_names)

    model = RNNPacked(vocab_size, dim_embeddings, hidden_nodes, n_classes)

    loss_func = nn.CrossEntropyLoss()

    train_config = TrainingConfig(model=model, loss_func=loss_func, training_loader=data_loader_training, validation_loader=data_loader_validation)
    results_pd = train(train_config)  
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
    train_config = TrainingConfig(model=model, loss_func=loss_func, training_loader=data_loader_training, validation_loader=data_loader_validation)
    results_pd = train(train_config)     
    print(results_pd)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=10)[0].tolist())
    print(created_text)

def run_simpleGPT():
    corpus_file_training = 'data/small/training.txt'
    corpus_file_validation = 'data/small/validation.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    n_embd = 8
    num_heads = 4
    block_size = 5
    n_layer = 3
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
    model = simpleGPT(vocab_size=len(dataset_training.vocabulary),
                                   n_embd=n_embd,
                                   num_heads=num_heads,
                                   block_size=block_size,
                                   n_layer=n_layer, 
                                   dropout=dropout, 
                                   device=device)
    m = model.to(device)

    print(sum(p.numel() for p in m.parameters()), 'parameters')

    loss_func = cross_entropy_language_model
    train_config = TrainingConfig(model=model, loss_func=loss_func, training_loader=data_loader_training, validation_loader=data_loader_validation)
    results_pd = train(train_config)     
    print(results_pd)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=block_size*2, block_size=block_size)[0].tolist())
    print(created_text)

def run_GPT1():
    corpus_file_training = 'data/small/training.txt'
    corpus_file_validation = 'data/small/validation.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    config =  Config(vocab_size = 0,
            dim_embeddings = 8,
            dim_context = 5,
            num_heads = 4,
            n_layer = 2,
            dropout = 0.2,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    dataset_training = LanguageModelDataset(corpus_file_training, block_size=config.dim_context)
    data_loader_training = DataLoader(dataset_training, batch_size=10, shuffle=True, generator=generator)
    dataset_validation= LanguageModelDataset(corpus_file_validation,
                                             block_size=config.dim_context,
                                             vocabulary=dataset_training.vocabulary,
                                             encoder=dataset_training.encoder,
                                             decoder=dataset_training.decoder)
    data_loader_validation = DataLoader(dataset_validation, batch_size=10, shuffle=True, generator=generator)

    config.vocab_size = len(dataset_training.vocabulary)

    model = GPT1(config)
    m = model.to(config.device)

    print(sum(p.numel() for p in m.parameters()), 'parameters')

    loss_func = cross_entropy_language_model
    train_config = TrainingConfig(model=model, loss_func=loss_func, training_loader=data_loader_training, validation_loader=data_loader_validation)
    results_pd = train(train_config)     
    print(results_pd)

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=config.dim_context*2, block_size=config.dim_context)[0].tolist())
    print(created_text)

def run_GPT2():
    corpus_file_training = 'data/small/training.txt'
    corpus_file_validation = 'data/small/validation.txt'

    generator = torch.Generator()
    generator.manual_seed(5)

    config =  Config(vocab_size = 0,
            dim_embeddings = 8,
            dim_context = 5,
            num_heads = 4,
            n_layer = 2,
            dropout = 0.2,
            bias = True,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    dataset_training = LanguageModelDataset(corpus_file_training, block_size=config.dim_context)
    data_loader_training = DataLoader(dataset_training, batch_size=10, shuffle=True, generator=generator)
    dataset_validation= LanguageModelDataset(corpus_file_validation,
                                             block_size=config.dim_context,
                                             vocabulary=dataset_training.vocabulary,
                                             encoder=dataset_training.encoder,
                                             decoder=dataset_training.decoder)
    data_loader_validation = DataLoader(dataset_validation, batch_size=10, shuffle=True, generator=generator)

    config.vocab_size = len(dataset_training.vocabulary)

    model = GPT2(config)
    m = model.to(config.device)

    print(sum(p.numel() for p in m.parameters()), 'parameters')

    loss_func = cross_entropy_language_model
    train_config = TrainingConfig(model=model, loss_func=loss_func, training_loader=data_loader_training, validation_loader=data_loader_validation)
    results_pd = train(train_config)     
    print(results_pd)

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    created_text = dataset_training.decoder(generate(m, context, max_new_tokens=config.dim_context*2, block_size=config.dim_context)[0].tolist())
    print(created_text)

# MNIST ATTENTION

def train_baseline(): 
    try: 
        mnist_train = torchvision.datasets.MNIST("data/", train=True, transform=transforms.ToTensor(), download=False)
        mnist_validation = torchvision.datasets.MNIST("data/", train=False, transform=transforms.ToTensor(), download=False)
    except:
        mnist_train = torchvision.datasets.MNIST("data/", train=True, transform=transforms.ToTensor(), download=True)
        mnist_validation  = torchvision.datasets.MNIST("data/", train=False, transform=transforms.ToTensor(), download=True)



    B = 2

    largest_training = LargestDigit(mnist_train)
    largest_validation = LargestDigit(mnist_validation)

    training_loader = DataLoader(largest_training, batch_size=B, shuffle=True)
    validation_loader = DataLoader(largest_validation, batch_size=B)

    H = 5
    classes = 10
    simpleNet = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784*3,H), # 784*3 because there are 784 pixels in an image and 3 images in the bag
        nn.LeakyReLU(),
        nn.BatchNorm1d(H),
        nn.Linear(H,H),
        nn.LeakyReLU(),
        nn.BatchNorm1d(H),
        nn.Linear(H,H),
        nn.LeakyReLU(),
        nn.BatchNorm1d(H),
        nn.Linear(H, classes)
    ) 

    train_config = TrainingConfig(model=simpleNet,
                                    loss_func=nn.CrossEntropyLoss(),
                                    training_loader=training_loader,
                                    validation_loader=validation_loader,
                                    save_model=True,
                                    save_path=os.path.join(os.getcwd(), "runs", "attention"),
                                    model_name="simple")
    result = train(train_config)


    print("done")

def train_simple_attention():
    try: 
        mnist_train = torchvision.datasets.MNIST("data/", train=True, transform=transforms.ToTensor(), download=False)
        mnist_validation = torchvision.datasets.MNIST("data/", train=False, transform=transforms.ToTensor(), download=False)
    except:
        mnist_train = torchvision.datasets.MNIST("data/", train=True, transform=transforms.ToTensor(), download=True)
        mnist_validation  = torchvision.datasets.MNIST("data/", train=False, transform=transforms.ToTensor(), download=True)

    B = 2

    largest_training = LargestDigit(mnist_train)
    largest_validation = LargestDigit(mnist_validation)

    training_loader = DataLoader(largest_training, batch_size=B, shuffle=True)
    validation_loader = DataLoader(largest_validation, batch_size=B)

    D = 28*28
    H = 5
    classes = 10

    # Feature extraction 
    backboneNetwork = nn.Sequential(
        Flatten2(),# Shape is now (B, T, D)
        nn.Linear(D,H), #Shape becomes (B, T, H)
        nn.LeakyReLU(),
        nn.Linear(H,H),
        nn.LeakyReLU(),
        nn.Linear(H,H),
        nn.LeakyReLU(), #still (B, T, H) on the way out
    )

    # Weights
    attentionMechanism = nn.Sequential(
    #Shape is (B, T, H)
    nn.Linear(H,H),
    nn.LeakyReLU(),
    nn.Linear(H, 1), # (B, T, 1)
    nn.Softmax(dim=1),
    )

    simpleAttentionNet = nn.Sequential(
        #input is (B, T, C, W, H). backbone & attention will be used by combiner to process
        Combiner(backboneNetwork, attentionMechanism), # result is (B, H)
        nn.BatchNorm1d(H),
        nn.Linear(H,H),
        nn.LeakyReLU(),
        nn.BatchNorm1d(H),
        nn.Linear(H, classes)
    )

    train_config = TrainingConfig(model=simpleAttentionNet,
                                  loss_func=nn.CrossEntropyLoss(),
                                  training_loader=training_loader,
                                  validation_loader=validation_loader,
                                  save_model=True,
                                  save_path=os.path.join(os.getcwd(), "runs", "attention"),
                                  model_name="simple")
    result = train(train_config)


    print("done")

def train_mnist_attention():
    try: 
        mnist_train = torchvision.datasets.MNIST("data/", train=True, transform=transforms.ToTensor(), download=False)
        mnist_validation = torchvision.datasets.MNIST("data/", train=False, transform=transforms.ToTensor(), download=False)
    except:
        mnist_train = torchvision.datasets.MNIST("data/", train=True, transform=transforms.ToTensor(), download=True)
        mnist_validation  = torchvision.datasets.MNIST("data/", train=False, transform=transforms.ToTensor(), download=True)

    B = 10

    largest_train = LargestDigitVariable(mnist_train)
    largest_validation = LargestDigitVariable(mnist_validation)
    training_loader = DataLoader(largest_train, batch_size=B, shuffle=True)
    validation_loader = DataLoader(largest_validation, batch_size=B)



    D = 28*28
    H = 5
    classes = 10

    model = SmarterAttentionNet(D, H, classes)
    train_config = TrainingConfig(model=model,
                                  loss_func=nn.CrossEntropyLoss(),
                                  training_loader=training_loader,
                                  validation_loader=validation_loader,
                                  save_model=True,
                                  save_path=os.path.join(os.getcwd(), "runs", "attention"),
                                  model_name="simple")
    result = train(train_config)

    print(result)


    print("done")


if __name__ == "__main__": 
    feadforward_moon()
