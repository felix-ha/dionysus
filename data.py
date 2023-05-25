import requests, zipfile, io
import unicodedata
import string
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_distinct_characters(text: str) -> list[str]:
    """
    Returns all unique charachters of the text as a list. 
    """
    return sorted(list(set(text)))

def create_token_index_mappings(vocabulary: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Creates for a given list of distinct tokens, i. e. the vocabulary, 
    dictionaries for token to index and index to token mappings. 
    The index is given via the position in the list. 
    """
    token_to_index = { token:index for index, token in enumerate(vocabulary) }
    index_to_token = { index:token for index, token in enumerate(vocabulary) }
    return token_to_index, index_to_token

def create_encoder(token_to_index):
    """
    Creates a function that converts a string into a list of integers,
    that represent the indexes of the tokens given the token_to_index dictionary. 
    """
    return lambda string: [token_to_index[char] for char in string]

def create_decoder(index_to_token):
    """
    Creates a function that converts a list of integers that represent the indexes
    of the tokens given the token_to_index dictionary and converts it to a string. 
    """
    return lambda idxs: ''.join([index_to_token[index] for index in idxs])

def create_corpus_index(corpus_raw: str):
    """
    Given a string (corpus_raw), one dimensional tensor with the indexes
    wrt the vocabulary based on this string is created.
    """
    vocabulary = get_distinct_characters(corpus_raw)
    token_to_index, index_to_token = create_token_index_mappings(vocabulary)
    encoder = create_encoder(token_to_index)
    decoder = create_decoder(index_to_token)
    corpus_index = torch.tensor(encoder(corpus_raw), dtype=torch.long)
    return corpus_index, vocabulary, decoder, encoder

def create_train_val_split(corpus_index: torch.Tensor, validation_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the corpus_index into training and validation set.
        The first (1-validation_ratio) % tokens will be the training data and 
        last validation_ratio % tokens the validation data.
        """
        n = int(validation_ratio*len(corpus_index)) 
        corpus_index_training = corpus_index[:n]
        corpus_index_validation = corpus_index[n:]
        return corpus_index_training,corpus_index_validation

def get_batch(data, block_size, batch_size):
    """
    Creates a batch with batch_size sequences, x is a sequence of length block_size
    y is a sequence of length block_size shifted one index to the right. 
    """
    max_index = len(data) - block_size
    ix = torch.randint(high=max_index, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class LanguageModelDataset(Dataset):
    def __init__(self, text_file, block_size, vocabulary=None, encoder=None, decoder=None):
        self.block_size = block_size
        with open(text_file, 'r', encoding='utf-8') as f:
            self.corpus_raw = f.read()
        if vocabulary is None or decoder is None or encoder is None:
            self.corpus_index, self.vocabulary, self.decoder, self.encoder = create_corpus_index(self.corpus_raw)
        else:
            self.vocabulary = vocabulary
            self.encoder = encoder
            self.decoder = decoder
            self.corpus_index = torch.tensor(encoder(self.corpus_raw), dtype=torch.long)

    def __len__(self):
        # last elements must not be collected, because they have no successor
        return len(self.corpus_index)-self.block_size

    def __getitem__(self, idx):
        return self.corpus_index[idx:idx+self.block_size],self.corpus_index[idx+1:idx+self.block_size+1]
    


def unicodeToAscii(s, all_letters):
    """
    Turns a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


class LanguageNameDataset(Dataset):
    
    def __init__(self):
        data_dir = Path("data/names")
        zip_file_url = "https://download.pytorch.org/tutorial/data.zip"
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))

        if not data_dir.exists():
            z.extractall()

        all_letters = string.ascii_letters + " .,;'"
        self.vocab_size = len(all_letters)
        n_letters = len(all_letters)
        self.token_to_index = {}
        for i in range(n_letters):
            self.token_to_index[all_letters[i]] = i

        name_language_data ={}
        for zip_path in [str(p).replace("\\", "/") for p in data_dir.iterdir()]:
            if zip_path.endswith(".txt"):
                lang = zip_path[len("data/names/"):-len(".txt")]
                with z.open(zip_path) as myfile:
                    lang_names = [unicodeToAscii(line, all_letters).lower() for line in str(myfile.read(), encoding='utf-8').strip().split("\n")]
                    name_language_data[lang] = lang_names

        self.label_names = [x for x in name_language_data.keys()]
        self.data = []
        self.labels = []
        for y, language in enumerate(self.label_names):
            for sample in name_language_data[language]:
                self.data.append(sample)
                self.labels.append(y)
        
    def __len__(self):
        return len(self.data)
    
    def string2InputVec(self, input_string):
        """
        This method will convert any input string into a vector of long values, according to the vocabulary used by this object. 
        input_string: the string to convert to a tensor
        """
        T = len(input_string) #How many characters long is the string?
        
        #Create a new tensor to store the result in
        name_vec = torch.zeros((T), dtype=torch.long)
        #iterate through the string and place the appropriate values into the tensor
        for pos, character in enumerate(input_string):
            name_vec[pos] = self.token_to_index[character]
            
        return name_vec
    
    def __getitem__(self, idx):
        name = self.data[idx]
        label = self.labels[idx]
        
        #Conver the correct class label into a tensor for PyTorch
        label_vec = torch.tensor([label], dtype=torch.long)
        
        return self.string2InputVec(name), label


class LargestDigit(Dataset):
    """
    Creates a modified version of a dataset where some number of samples are taken, 
    and the true label is the largest label sampled. When used with MNIST the labels 
    correspond to their values (e.g., digit "6" has label 6)
    """

    def __init__(self, dataset, toSample=3):
        """
        dataset: the dataset to sample from
        toSample: the number of items from the dataset to sample
        """
        self.dataset = dataset
        self.toSample = toSample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        #Randomly select n=self.toSample items from the dataset
        selected = np.random.randint(0,len(self.dataset), size=self.toSample)
        
        #Stack the n items of shape (B, *) shape into (B, n, *)
        x_new = torch.stack([self.dataset[i][0] for i in selected])
        #Label is the maximum label
        y_new = max([self.dataset[i][1] for i in selected])
        #Return (data, label) pair!
        return x_new, y_new
    

class LargestDigitVariable(Dataset):
    """
    Creates a modified version of a dataset where some variable number of samples are 
    taken, and the true label is the largest label sampled. When used with MNIST the
    labels correspond to their values (e.g., digit "6" has label 6). Each datum will 
    be padded with 0 values if the maximum number of items was not sampled. 
    """

    def __init__(self, dataset, maxToSample=6):
        """
        dataset: the dataset to sample from
        toSample: the number of items from the dataset to sample
        """
        self.dataset = dataset
        self.maxToSample = maxToSample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        #NEW: how many items should we select?
        how_many = np.random.randint(1,self.maxToSample, size=1)[0]
        #Randomly select n=self.toSample items from the dataset
        selected = np.random.randint(0,len(self.dataset), size=how_many)
        
        #Stack the n items of shape (B, *) shape into (B, n, *)
        #NEW: pad with zero values up to the max size
        x_new = torch.stack([self.dataset[i][0] for i in selected] + 
                            [torch.zeros((1,28,28)) for i in range(self.maxToSample-how_many)])
        #Label is the maximum label
        y_new = max([self.dataset[i][1] for i in selected])
        #Return (data, label) pair
        return x_new, y_new


if __name__ == "__main__":

    dataset = LanguageModelDataset('data/text.txt', block_size=10)
    dataloader = DataLoader(dataset, batch_size=1)

    for x, y in dataloader:
        print(f"{x}, {y}")
