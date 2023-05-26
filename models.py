import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    vocab_size:int
    dim_embeddings: int
    dim_context: int
    num_heads: int
    n_layer: int
    dropout: int 
    bias: bool = True
    device: str = 'cpu'


def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj


class RNN(nn.Module):
    def __init__(self, vocab_size, dim_embeddings, hidden_nodes, n_classes):
        super().__init__()
        self.rnn = nn.Sequential(
                nn.Embedding(vocab_size, dim_embeddings), #(B, T) -> (B, T, D)
                nn.RNN(dim_embeddings, hidden_nodes, batch_first=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
                #the tanh activation is built into the RNN object, so we don't need to do it here
                LastTimeStep(), #We need to take the RNN output and reduce it to one item, (B, D)
                nn.Linear(hidden_nodes, n_classes), #(B, D) -> (B, classes)
                )

    def forward(self, x):
        logits = self.rnn(x)
        return logits
    

class RNNPacked(nn.Module):
    def __init__(self, vocab_size, dim_embeddings, hidden_nodes, n_classes):
        super().__init__()
        self.rnn = nn.Sequential(
                EmbeddingPackable(nn.Embedding(vocab_size, dim_embeddings)), #(B, T) -> (B, T, D)
                nn.RNN(dim_embeddings, hidden_nodes, batch_first=True), #(B, T, D) -> ( (B,T,D) , (S, B, D)  )
                #the tanh activation is built into the RNN object, so we don't need to do it here
                LastTimeStep(), #We need to take the RNN output and reduce it to one item, (B, D)
                nn.Linear(hidden_nodes, n_classes), #(B, D) -> (B, classes)
                )

    def forward(self, x):
        logits = self.rnn(x)
        return logits


class EmbeddingPackable(nn.Module):
    """
    The embedding layer in PyTorch does not support Packed Sequence objects. 
    This wrapper class will fix that. If a normal input comes in, it will 
    use the regular Embedding layer. Otherwise, it will work on the packed 
    sequence to return a new Packed sequence of the appropriate result. 
    """
    def __init__(self, embd_layer):
        super(EmbeddingPackable, self).__init__()
        self.embd_layer = embd_layer 

    def forward(self, input):
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            # We need to unpack the input, 
            sequences, lengths = torch.nn.utils.rnn.pad_packed_sequence(input.cpu(), batch_first=True)
            #Embed it
            sequences = self.embd_layer(sequences.to(input.data.device))
            #And pack it into a new sequence
            return torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.cpu(), 
                                                            batch_first=True, enforce_sorted=False)
        else:#apply to normal data
            return self.embd_layer(input)   


class LastTimeStep(nn.Module):
    """
    A class for extracting the hidden activations of the last time step following 
    the output of a PyTorch RNN module. 
    """
    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        if bidirectional:
            self.num_driections = 2
        else:
            self.num_driections = 1    
    
    def forward(self, input):
        #Result is either a tupe (out, h_t)
        #or a tuple (out, (h_t, c_t))
        rnn_output = input[0]
        last_step = input[1] #this will be h_t
        if(type(last_step) == tuple):#unless it's a tuple, 
            last_step = last_step[0]#then h_t is the first item in the tuple
        batch_size = last_step.shape[1] #per docs, shape is: '(num_layers * num_directions, batch, hidden_size)'
        #reshaping so that everything is separate 
        last_step = last_step.view(self.rnn_layers, self.num_driections, batch_size, -1)
        #We want the last layer's results
        last_step = last_step[self.rnn_layers-1] 
        #Re order so batch comes first
        last_step = last_step.permute(1, 0, 2)
        #Finally, flatten the last two dimensions into one
        return last_step.reshape(batch_size, -1)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        logits = self.token_embedding_table(idx) 
        return logits


class simpleGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, num_heads, block_size, n_layer, dropout, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=num_heads, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        positinal_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + positinal_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        return logits
    

class GPT1(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.dim_embeddings)
        self.position_embedding_table = nn.Embedding(config.dim_context, config.dim_embeddings)
        self.blocks = nn.Sequential(*[BlockGPT1(config.dim_embeddings, config.num_heads, config.dim_context, config.dropout) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.dim_embeddings, config.vocab_size)
        self.device = config.device

    def forward(self, idx):
        T = idx.shape[-1]
        embedding_token = self.token_embedding_table(idx) 
        embedding_position = self.position_embedding_table(torch.arange(T, device=self.device))
        x = embedding_token + embedding_position 
        x = self.blocks(x)
        logits = self.lm_head(x)
        return logits
    

class GPT2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.dim_embeddings)
        self.position_embedding_table = nn.Embedding(config.dim_context, config.dim_embeddings)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[BlockGPT2(config.dim_embeddings, config.num_heads, config.dim_context, config.bias, config.dropout) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.dim_embeddings, config.vocab_size)
        self.device = config.device

    def forward(self, idx):
        T = idx.shape[-1]
        embedding_token = self.token_embedding_table(idx) 
        embedding_position = self.position_embedding_table(torch.arange(T, device=self.device))
        x = self.drop(embedding_token + embedding_position)
        x = self.blocks(x)
        logits = self.lm_head(x)
        return logits


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
     

class FeedFowardGPT2(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[-2]
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BlockGPT1(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.multi_head = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = self.multi_head(x) + x
        x = self.ln1(x)
        x = self.ffwd(x) + x
        x = self.ln2(x)
        return x
    

class BlockGPT2(nn.Module):
    def __init__(self, n_embd, n_head, block_size, bias, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.multi_head = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFowardGPT2(n_embd, dropout)
        self.ln1 = LayerNorm(n_embd, bias)
        self.ln2 = LayerNorm(n_embd, bias)


    def forward(self, x):
        x = x + self.multi_head(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x)) 
        return x
   
    
def generate(model, idx, max_new_tokens, block_size=None):
    model.eval()
    for _ in range(max_new_tokens):
        if block_size is None:
            idx_cond = idx
        else: 
            idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1) 
        idx_next = torch.multinomial(probs, num_samples=1) 
        idx = torch.cat((idx, idx_next), dim=1) 
    return idx

# MNIST ATTENTION

class Flatten2(nn.Module):
    """
    Takes a vector of shape (A, B, C, D, E, ...)
    and flattens everything but the first two dimensions, 
    giving a result of shape (A, B, C*D*E*...)
    Creates the bag of digits for MNIST attention
    """
    def forward(self, input):
        return input.view(input.size(0), input.size(1), -1)
    

class Combiner(nn.Module):
    """
    This class is used to combine a feature exraction network F and a importance prediction network W,
    and combine their outputs by adding and summing them together. 
    """

    def __init__(self, featureExtraction, weightSelection):
        """
        featureExtraction: a network that takes an input of shape (B, T, D) and outputs a new 
            representation of shape (B, T, D'). 
        weightSelection: a network that takes in an input of shape (B, T, D') and outputs a 
            tensor of shape (B, T, 1) or (B, T). It should be normalized, so that the T 
            values at the end sum to one (torch.sum(_, dim=1) = 1.0)
        """
        super(Combiner, self).__init__()
        self.featureExtraction = featureExtraction
        self.weightSelection = weightSelection
    
    def forward(self, input):
        """
        input: a tensor of shape (B, T, D)
        return: a new tensor of shape (B, D')
        """
        features = self.featureExtraction(input) #(B, T, D) $\boldsymbol{h}_i = F(\boldsymbol{x}_i)$
        weights = self.weightSelection(features) #(B, T) or (B, T, 1) for $\boldsymbol{\alpha}$
        if len(weights.shape) == 2: #(B, T) shape
            weights.unsqueese(2) #now (B, T, 1) shape
        
        r = features*weights #(B, T, D), computes $\alpha_i \cdot \boldsymbol{h}_i$
        
        return torch.sum(r, dim=1) #sum over the T dimension, giving (B, D) final shape $\bar{\boldsymbol{x}}$
    

class DotScore(nn.Module):

    def __init__(self, H):
        """
        H: the number of dimensions coming into the dot score. 
        """
        super(DotScore, self).__init__()
        self.H = H
    
    def forward(self, states, context):
        """
        states: (B, T, H) shape
        context: (B, H) shape
        output: (B, T, 1), giving a score to each of the T items based on the context 
        
        """
        T = states.size(1)
        #compute $\boldsymbol{h}_t^\top \bar{\boldsymbol{h}}$
        scores = torch.bmm(states,context.unsqueeze(2)) / np.sqrt(self.H) #(B, T, H) -> (B, T, 1)
        return scores
  
    
# ToDo: Fix forward pass
class GeneralScore(nn.Module):

    def __init__(self, H):
        """
        H: the number of dimensions coming into the dot score. 
        """
        super(GeneralScore, self).__init__()
        self.w = nn.Bilinear(H, H, 1) #stores $W$
    
    def forward(self, states, context):
        """
        states: (B, T, H) shape
        context: (B, H) shape
        output: (B, T, 1), giving a score to each of the T items based on the context 
        
        """
        T = states.size(1)
        #Repeating the values T times 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, H) -> (B, T, H)
        #computes $\boldsymbol{h}_{t}^{\top} W \bar{\boldsymbol{h}}$
        scores = self.w(states, context) #(B, T, H) -> (B, T, 1)
        return scores 
    

# ToDo: Fix forward pass
class AdditiveAttentionScore(nn.Module):

    def __init__(self, H):
        super(AdditiveAttentionScore, self).__init__()
        self.v = nn.Linear(H, 1) 
        self.w = nn.Linear(2*H, H)#2*H because we are going to concatenate two inputs
    
    def forward(self, states, context):
        """
        states: (B, T, H) shape
        context: (B, H) shape
        output: (B, T, 1), giving a score to each of the T items based on the context 
        
        """
        T = states.size(1)
        #Repeating the values T times 
        context = torch.stack([context for _ in range(T)], dim=1) #(B, H) -> (B, T, H)
        state_context_combined = torch.cat((states, context), dim=2) #(B, T, H) + (B, T, H)  -> (B, T, 2*H)
        scores = self.v(torch.tanh(self.w(state_context_combined))) # (B, T, 2*H) -> (B, T, 1)
        return scores
    

class ApplyAttention(nn.Module):
    """
    This helper module is used to apply the results of an attention mechanism toa set of inputs. 
    Replaces combiner
    """

    def __init__(self):
        super(ApplyAttention, self).__init__()
        
    def forward(self, states, attention_scores, mask=None):
        """
        states: (B, T, H) shape giving the T different possible inputs
        attention_scores: (B, T, 1) score for each item at each context
        mask: None if all items are present. Else a boolean tensor of shape 
            (B, T), with `True` indicating which items are present / valid. 
            
        returns: a tuple with two tensors. The first tensor is the final context
        from applying the attention to the states (B, H) shape. The second tensor
        is the weights for each state with shape (B, T, 1). 
        """
        
        if mask is not None:
            #set everything not present to a large negative value that will cause vanishing gradients 
            attention_scores[~mask] = -1000.0
        #compute the weight for each score
        weights = F.softmax(attention_scores, dim=1) #(B, T, 1) still, but sum(T) = 1
    
        final_context = (states*weights).sum(dim=1) #(B, T, D) * (B, T, 1) -> (B, D)
        return final_context, weights
    

def getMaskByFill(x, time_dimension=1, fill=0):
    """
    x: the original input with three or more dimensions, (B, ..., T, ...)
        which may have unsued items in the tensor. B is the batch size, 
        and T is the time dimension. 
    time_dimension: the axis in the tensor `x` that denotes the time dimension
    fill: the constant used to denote that an item in the tensor is not in use,
        and should be masked out (`False` in the mask). 
    
    return: A boolean tensor of shape (B, T), where `True` indicates the value
        at that time is good to use, and `False` that it is not. 
    """
    to_sum_over = list(range(1,len(x.shape))) #skip the first dimension 0 because that is the batch dimension
    
    if time_dimension in to_sum_over:
        to_sum_over.remove(time_dimension)
        
    with torch.no_grad():
        #(x!=fill) determines locations that might be unused, beause they are 
        #missing the fill value we are looking for to indicate lack of use. 
        #We then count the number of non-fill values over everything in that
        #time slot (reducing changes the shape to (B, T)). If any one entry 
        #is non equal to this value, the item represent must be in use - 
        #so return a value of true. 
        mask = torch.sum((x != fill), dim=to_sum_over) > 0
    return mask


class SmarterAttentionNet(nn.Module):

    def __init__(self, input_size, hidden_size, out_size, score_net=None):
        super(SmarterAttentionNet, self).__init__()
        self.backbone = nn.Sequential(
            Flatten2(),# Shape is now (B, T, D)
            nn.Linear(input_size,hidden_size), #Shape becomes (B, T, H)
            nn.LeakyReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
        )#returns (B, T, H)
        
        #Try changing this and see how the results change!
        self.score_net = DotScore(hidden_size) if (score_net is None) else score_net

        self.apply_attn = ApplyAttention()
        
        self.prediction_net = nn.Sequential( #(B, H), 
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, out_size ) #(B, H)
        )
        
    
    def forward(self, input):

        mask = getMaskByFill(input)

        h = self.backbone(input) #(B, T, D) -> (B, T, H)

        #h_context = torch.mean(h, dim=1) 
        #computes torch.mean but ignoring the masked out parts
        #first add together all the valid items
        h_context = (mask.unsqueeze(-1)*h).sum(dim=1)#(B, T, H) -> (B, H)
        #then divide by the number of valid items, pluss a small value incase a bag was all empty
        h_context = h_context/(mask.sum(dim=1).unsqueeze(-1)+1e-10)

        scores = self.score_net(h, h_context) # (B, T, H) , (B, H) -> (B, T, 1)

        final_context, _ = self.apply_attn(h, scores, mask=mask)

        return self.prediction_net(final_context)
    

class Seq2SeqAttention(nn.Module):

    def __init__(self, num_embeddings, embd_size, hidden_size, padding_idx=None, layers=1, max_decode_length=20):
        super(Seq2SeqAttention, self).__init__()
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.embd = nn.Embedding(num_embeddings, embd_size, padding_idx=padding_idx)
        
        #We set the hidden size to half the intended length, because we will make the 
        #encoder bi-directional. That means we will get 2 hidden state representations
        #which we will concatinate together, giving us the desired size!
        self.encode_layers = nn.GRU(input_size=embd_size, hidden_size=hidden_size//2, 
                                       num_layers=layers, bidirectional=True)
        #decoder will be uni-directionall, and we need to use CRUCells so that we can 
        #do the decoding one step at a time
        self.decode_layers = nn.ModuleList([nn.GRUCell(embd_size, hidden_size)] + 
                                     [nn.GRUCell(hidden_size, hidden_size) for i in range(layers-1)])
        self.score_net = DotScore(hidden_size)
        #predict_word will be a small fully connected network that we use to convert the 
        #result of the attention mechanism and the local context into a prediction for 
        #the next word
        self.predict_word = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_embeddings)
        )
        self.max_decode_length = max_decode_length
        self.apply_attn = ApplyAttention()
    
    def forward(self, input):
        #Input should be (B, T) or ((B, T), (B, T'))
        if isinstance(input, tuple):
            input, target = input
        else:
            target = None
        #What is the batch size?
        B = input.size(0)
        #What is the max number of input time steps?
        T = input.size(1)

        x = self.embd(input) #(B, T, D)

        #grab the device that the model currently resides on
        #we will need this later 
        device = x.device

        mask = getMaskByFill(x) 

        #We will use the mask to figure out how long 
        #each input sequence is
        seq_lengths = mask.sum(dim=1).view(-1) #shape (B), containing the # of non-zero values
        #the sequence lengths will be used to create a packed input for the encoder RNN
        x_packed = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        h_encoded, h_last = self.encode_layers(x_packed)
        h_encoded, _ = pad_packed_sequence(h_encoded) #(B, T, 2, D//2) , b/c its bidirectional
        h_encoded = h_encoded.view(B, T, -1) #(B, T, D)
        #now h_encoded is the result of running the encoder RNN on the input!


        #getting the last hidden state is a little trickier
        #first the output gets reshaped as (num_layers, directions, batch_size, hidden_size)
        #and then we grab the last index in the first dimension, because we want the 
        #last layer's output
        hidden_size = h_encoded.size(2) 
        h_last = h_last.view(-1, 2, B, hidden_size//2)[-1,:,:,:] #shape is now (2, B, D/2)
        #now we will reorder to (B, 2, D/2), and flatten the last two dimensions down to (B, D)
        h_last = h_last.permute(1, 0, 2).reshape(B, -1)
        
        
        #End of Encoding portion. h_encoded now contains the representation of the input data!
        #h_last has the final ouputs of the RNN, to use as the initial input state for the decoder
        
        #The first input to the decoder will be the output of the last encoder step
        #decoder_input = h_last
        
        # new hidden states for decoders
        h_prevs = [h_last for l in range(len(self.decode_layers))]

        #We will save all the attention mechanism results for visualization later!
        all_attentions = []
        all_predictions = []

        #Grab the last item from the input (which should be an EOS marker)
        #as the first input for the decoder
        #We could also hard-code the SOS marker instead
        decoder_input = self.embd(input.gather(1,seq_lengths.view(-1,1)-1).flatten()) #(B, D)

        #How many decoding steps should we do?
        steps = min(self.max_decode_length, T)
        #If we are training, the target values tells us exactly
        #how many steps to take
        if target is not None: #We know the exact decode length!
            steps = target.size(1)
        
        #Do we use teacher forcing (true) or auto-regressive (false)
        teacher_forcing = np.random.choice((True,False))
        for t in range(steps):
            x_in = decoder_input #(B, D)

            for l in range(len(self.decode_layers)):
                h_prev = h_prevs[l] 
                h = self.decode_layers[l](x_in, h_prev)

                h_prevs[l] = h
                x_in = h
            h_decoder = x_in #(B, D), we now have the hidden state for the decoder at this time step

            #This is the attention mechanism, lets look at all the previous encoded states and 
            #see which look relevant

            scores = self.score_net(h_encoded, h_decoder) #(B, T, 1)
            context, weights = self.apply_attn(h_encoded, scores, mask=mask)

            #save the attention weights for visualization later
            all_attentions.append( weights.detach() ) #we are detaching the weights because we 
            #do not want to compute anything with them anymore, we just want to save their 
            #values to make visualizations

            #Now lets compute the final representation by concatinating the 
            #attention result and the initial context
            word_pred = torch.cat((context, h_decoder), dim=1) #(B, D) + (B, D)  -> (B, 2*D)
            #and get a prediction about what the next token is by pushing it
            #through a small fully-connected network
            word_pred = self.predict_word(word_pred) #(B, 2*D) -> (B, V)
            all_predictions.append(word_pred)
    
            #Now we have $\hat{y}_t$! we need to select the input for the next
            #time step. We use torch.no_grad() because the gradient will 
            #carry through the hidden states of the RNN, not the input tokens
            with torch.no_grad():
                if self.training:
                    if target is not None and teacher_forcing:
                        #We have the target and selected teacher forcing, so use the
                        #correct next answer
                        next_words = target[:,t].squeeze()
                    else:
                        #Sample the next token based on the predictions made
                        next_words = torch.multinomial(F.softmax(word_pred, dim=1), 1)[:,-1]
                else:
                    #we are trying to make an actual prediction, so take the most likely word
                    #we could improve this by using temperature and sampling like we did 
                    #for the CharRNN model!
                    next_words = torch.argmax(word_pred, dim=1)
            #end of torch.no_grad()
            
            #We've decided what the next tokens are, we are back to using
            #the gradient calculation so that the embedding layer is adjusted
            #appropriately during training. 
            decoder_input = self.embd(next_words.to(device))
    
        #done decoding!
        if self.training: #When training, only the predictions are important
            return torch.stack(all_predictions, dim=1)
        else:#When evaluatin, we also want to look at the attention weights
            return torch.stack(all_predictions, dim=1), torch.stack(all_attentions, dim=1).squeeze()


def CrossEntLossTime(x, y, word2indx, PAD_token):
    """
    x: output with shape (B, T, V)
    y: labels with shape (B, T')
    
    """
    if isinstance(x, tuple):
        x, _ = x
    #We do not want to compute a loss for items that have been padded out!
    cel = nn.CrossEntropyLoss(ignore_index=word2indx[PAD_token])
    T = min(x.size(1), y.size(1))
    
    loss = 0
    for t in range(T):
        loss += cel(x[:,t,:], y[:,t])
    return loss


