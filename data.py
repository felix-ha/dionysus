import requests, zipfile, io
import unicodedata
import string
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import re
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter 
from torchtext.vocab import vocab 

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


def pad_and_pack(batch):
    #1, 2, & 3: organize the batch input lengths, inputs, and outputs as seperate lists
    input_tensors = []
    labels = []
    lengths = []
    for x, y in batch:
        input_tensors.append(x)
        labels.append(y)
        lengths.append(x.shape[0]) #Assume shape is (T, *)
    #4: create the padded version of the input
    x_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
    #5: create the packed version from the padded & lengths
    x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)
    #Convert the lengths into a tensor
    y_batched = torch.as_tensor(labels, dtype=torch.long)
    #6: return a tuple of the packed inputs and their labels
    return x_packed, y_batched


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


def download_prepare_and_save_eng_fra(pkl_file):
    all_data = []
    resp = urlopen("https://download.pytorch.org/tutorial/data.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    for line in zipfile.open("data/eng-fra.txt").readlines():
        line = line.decode('utf-8').lower()#lower case only please
        line = re.sub(r"[-.!?]+", r" ", line)#no puntuation
        source_lang, target_lang = line.split("\t")[0:2]
        all_data.append( (source_lang.strip(), target_lang.strip()))

    with open(pkl_file, 'wb') as file:
        pickle.dump(all_data, file)

def load_raw_eng_fra(pkl_file):
    with open(pkl_file, 'rb') as file:
        return pickle.load(file)


class TranslationDataset(Dataset):
    """
    Takes a dataset with tuples of strings (x, y) and
    converts them to tuples of int64 tensors. 
    This makes it easy to encode Seq2Seq problems.
    
    Strings in the input and output targets will be broken up by spaces
    """

    def __init__(self, pkl_file, MAX_LEN, SOS_token, EOS_token, PAD_token):
        """
        lang_pairs: a List[Tuple[String,String]] containing the source,target pairs for a Seq2Seq problem. 
        word2indx: a Map[String,Int] that converts each word in an input string into a unique ID. 
        """
        try:
            all_data = load_raw_eng_fra(pkl_file) 
        except:
            download_prepare_and_save_eng_fra(pkl_file)
            all_data = load_raw_eng_fra(pkl_file) 


        short_subset = [] #the subset we will actually use
        for (s, t) in all_data:
            if max(len(s.split(" ")), len(t.split(" "))) <= MAX_LEN:
                short_subset.append((s,t))
        print("Using ", len(short_subset), "/", len(all_data))

        self.SOS_token = SOS_token
        self.EOS_token = EOS_token 
        self.PAD_token = PAD_token

        # Words from the source and target language are thrown in the same vocabulary
        word2indx = {self.PAD_token:0, self.SOS_token:1, self.EOS_token:2}
        for s, t in short_subset:
            for sentance in (s, t):
                for word in sentance.split(" "):
                    if word not in word2indx:
                        word2indx[word] = len(word2indx)
        print("Size of Vocab: ", len(word2indx))
        #build the inverted dict for looking at the outputs later
        indx2word = {}
        for word, indx in word2indx.items():
            indx2word[indx] = word

        self.lang_pairs = short_subset
        self.word2indx = word2indx
        self.indx2word = indx2word

    def __len__(self):
        return len(self.lang_pairs)

    def __getitem__(self, idx):
        x, y = self.lang_pairs[idx]
        x = self.SOS_token + " " + x + " " + self.EOS_token
        y = y + " " + self.EOS_token
        
        #convert to lists of integers
        x = [self.word2indx[w] for w in x.split(" ")]
        y = [self.word2indx[w] for w in y.split(" ")]
        
        x = torch.tensor(x, dtype=torch.int64)
        y = torch.tensor(y, dtype=torch.int64)
        
        return x, y
    

def pad_batch_seq2seq(batch, word2indx, PAD_token):
    """
    Pad items in the batch to the length of the longest item in the batch
    """
    #We actually have two different maxiumum lengths! The max length of the input sequences, and the max 
    #length of the output sequences. So we will determine each seperatly, and only pad the inputs/outputs
    #by the exact amount we need
    max_x = max([i[0].size(0) for i in batch])
    max_y = max([i[1].size(0) for i in batch])
    
    PAD = word2indx[PAD_token]
    
    #We will use the F.pad function to pad each tensor to the right
    X = [F.pad(i[0], (0,max_x-i[0].size(0)), value=PAD) for i in batch]
    Y = [F.pad(i[1], (0,max_y-i[1].size(0)), value=PAD) for i in batch]
    
    X, Y = torch.stack(X), torch.stack(Y)
    
    return (X, Y), Y


def text_transform(x, vocab, tokenizer): #string -> list of integers
    return [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']] #vocab acts like a dictionary, handls unkown tokens for us, and we can make it pre and post-pend with the start and end markers respectively.

def label_transform(x): 
    return x-1 #labes are originally [1, 2, 3, 4] but we need them as [0, 1, 2, 3] 

def pad_batch(batch, vocab, tokenizer, padding_idx):
    """
    Pad items in the batch to the length of the longest item in the batch. 
    Also, re-order so that the values are returned (input, label)
    """
    labels = [label_transform(z[0]) for z in batch] #get and transform every label in the batch
    texts = [torch.tensor(text_transform(z[1], vocab, tokenizer), dtype=torch.int64) for z in batch] #get, tokenizer, and put into a tensor every text
    #what is the longest sequence in this batch? 
    max_len = max([text.size(0) for text in texts])
    #pad each text tensor by whatever amount gets it to the max_len
    texts = [F.pad(text, (0,max_len-text.size(0)), value=padding_idx) for text in texts]
    #make x and y a single tensor
    x, y = torch.stack(texts), torch.tensor(labels, dtype=torch.int64)
    
    return x, y


def get_ag_news_dataloaders(B, min_freq):
    pkl_file = "data.pkl"
    unk_token = '<UNK>'
    padding_token = '<PAD>'
    begin_sentence_token = '<BOS>'
    end_sentence_token  = '<EOS>'
    special_tokens=(unk_token, begin_sentence_token, end_sentence_token, padding_token)

    try:
        with open(pkl_file, 'rb') as file:
            data = pickle.load(file)
            train_dataset = data['train_dataset']
            test_dataset = data['test_dataset']
    except:
        # TODO: write raw data into tempdirectory
        train_iter, test_iter = AG_NEWS(root='./data', split=('train', 'test'))
        train_dataset = list(train_iter)
        test_dataset = list(test_iter)

        data = {'train_dataset': train_dataset, 'test_dataset': test_dataset}
        with open(pkl_file, 'wb') as file:
            pickle.dump(data, file)

    # TODO remove subset 
    train_dataset = train_dataset[:500]
    print("WARNING: ONLY A SUBSET OF THE TRAININGSDATA IS USED!!!")

    # TODO: Improve tokenzer -> for example remove "."
    tokenizer = get_tokenizer('basic_english')
    counter = Counter() 
    for (label, line) in train_dataset: #loop through the training data 
        counter.update(tokenizer(line)) #count the number of unique tokens we see and how often we see them (e.g., we will see "the" a lot, but "sasquatch" maybe once or not at all.)
    vocabulary = vocab(counter, min_freq=min_freq, specials=special_tokens) #create a vocab object, removing any word that didn't occur at least 10 times, and add special vocab items for unkown, begining of sentance, end of sentance, and "padding"
    vocabulary.set_default_index(vocabulary[unk_token])

    text = text_transform(f"{unk_token} this is new halloasdf", vocabulary, tokenizer)
    # assert text == [1, 0, 678, 165, 92, 0, 2]
    padding_idx = vocabulary[padding_token]
    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, collate_fn=lambda x: pad_batch(x, vocabulary, tokenizer, padding_idx))
    test_loader = DataLoader(test_dataset, batch_size=B, collate_fn=lambda x: pad_batch(x, vocabulary, tokenizer, padding_idx))

    NUM_CLASS = len(np.unique([z[0] for z in train_dataset])) 
    return train_loader, test_loader, NUM_CLASS, vocabulary, padding_idx


if __name__ == "__main__":
    pass
