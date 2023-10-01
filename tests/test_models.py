import unittest
import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dionysus.data import *
from src.dionysus.models import *
from src.dionysus.training import TrainingConfig, train
from src.dionysus.constants import CHECKPOINT_FILE

class TestSanityChecks(unittest.TestCase):
    def test_RNN(self):
        B, T, D = 2, 5, 2
        H = 3
        x = torch.rand([B, T, D])
        h_0 = torch.zeros([1, B, H]) # 1 is for number of layers, here 1

        rnn = nn.RNN(D, H, batch_first=True)
        output, h_n = rnn(x, h_0)

        self.assertTrue(output.shape == (B, T, H))
        self.assertTrue(h_n.shape == (1, B, H))

    def test_GRU(self):
        B, T, D = 2, 5, 2
        H = 3
        x = torch.rand([B, T, D])
        h_0 = torch.zeros([1, B, H]) # 1 is for number of layers, here 1

        rnn = nn.GRU(D, H, batch_first=True)
        output, h_n = rnn(x, h_0)

        self.assertTrue(output.shape == (B, T, H))
        self.assertTrue(h_n.shape == (1, B, H))





class TestBigramLanguageModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
      self.vocab_size=26

    def test_single_token_input_bigram(self):
        x = torch.tensor([1])

        model = BigramLanguageModel(vocab_size=self.vocab_size)
                  
        logits = model(x)
        self.assertTrue(logits.shape == (1, self.vocab_size))

    def test_single_token_input_bigram2(self):
        x = torch.tensor([[1]])

        model = BigramLanguageModel(vocab_size=self.vocab_size)
                  
        logits = model(x)
        self.assertTrue(logits.shape == (1, 1, self.vocab_size))

    def test_single_token_input_gpt1(self):
        x = torch.tensor([1])

        model =  simpleGPT(vocab_size=self.vocab_size,
                                   n_embd=5,
                                   num_heads=1,
                                   block_size=500,
                                   n_layer=1, 
                                   dropout=0.2, 
                                   device='cpu')
        # ToDo: Find better way to implement positional encoding
        with self.assertRaises(ValueError):
            logits = model(x)

    def test_single_token_input_gpt(self):
        x = torch.tensor([[1]])

        model =  simpleGPT(vocab_size=self.vocab_size,
                                   n_embd=5,
                                   num_heads=1,
                                   block_size=500,
                                   n_layer=1, 
                                   dropout=0.2, 
                                   device='cpu')


        logits = model(x)
        self.assertTrue(logits.shape == (1, 1, self.vocab_size))
    
    def test_single_batch_loop(self):
        x = torch.tensor([[1,2]])

        models = [BigramLanguageModel(vocab_size=self.vocab_size), 
                  simpleGPT(vocab_size=self.vocab_size,
                                   n_embd=5,
                                   num_heads=1,
                                   block_size=2,
                                   n_layer=1, 
                                   dropout=0.2, 
                                   device='cpu')]

        for model in models:
            logits = model(x)
            B, T, C = logits.shape
            self.assertTrue(B == 1) # B_x = 1
            self.assertTrue(T == 2) # T_x = 2
            self.assertTrue(C == self.vocab_size) # Channels are logits that yield into a distribution for the vocabulary

    def test_several_batches(self):
        x = torch.tensor([[1, 2, 3, 4, 5],
                          [4, 23, 1, 0, 25],
                          [22, 2, 4, 12, 19]]) 
        
        models = [BigramLanguageModel(vocab_size=self.vocab_size), 
                  simpleGPT(vocab_size=self.vocab_size,
                                   n_embd=5,
                                   num_heads=1,
                                   block_size=5,
                                   n_layer=1, 
                                   dropout=0.2, 
                                   device='cpu')]

        for model in models:
            logits = model(x)
            B, T, C = logits.shape
            self.assertTrue(B == 3) # B_x = 1
            self.assertTrue(T == 5) # T_x = 5
            self.assertTrue(C == self.vocab_size)

    def test_embedding(self):
        x = torch.tensor([[1, 2, 3, 4, 5],
                        [4, 23, 1, 0, 25],
                        [22, 2, 4, 12, 19]]) 
        
        B, T = x.shape
        self.assertTrue(B == 3)
        self.assertTrue(T == 5)
        
        vocab_size = torch.max(x).item() + 1
        self.assertTrue(vocab_size == 26) 

        n_embd = 50
        embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)

        out = embedding(x)

        self.assertTrue(out.shape == (B, T, n_embd))

    def test_self_attention_head(self):
        B = 1
        n_embd = 26
        block_size = 5 # This is T
        head_size = 3
        x = torch.rand((B, block_size, n_embd))
              
        head = Head(head_size, n_embd, block_size)
        out = head(x)

        self.assertTrue(out.shape == (B, block_size, head_size))

    def test_self_attention_multi_head(self):
        B = 1
        n_embd = 8
        block_size = 5 # This is T
        head_size = 2
        x = torch.rand((B, block_size, n_embd))
        
        num_heads = 4
        head = MultiHeadAttention(num_heads, head_size, n_embd, block_size)
        out = head(x)

        self.assertTrue(out.shape == (B, block_size, num_heads * head_size))

    def test_embedding_self_attention(self):
        x = torch.tensor([[1, 2, 3, 4, 5],
                        [4, 23, 1, 0, 25],
                        [22, 2, 4, 12, 19]]) 
        
        B, T = x.shape
        self.assertTrue(B == 3)
        self.assertTrue(T == 5)
        
        vocab_size = torch.max(x).item() + 1
        self.assertTrue(vocab_size == 26) 

        n_embd = 50
        head_size = 25
        block_size = T
        embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        head = Head(head_size, n_embd, block_size)

        emb = embedding(x)
        out = head(emb)

        self.assertTrue(emb.shape == (B, T, n_embd))
        self.assertTrue(out.shape == (B, T, head_size))

    def test_nn_linear_no_batch(self):
        with torch.no_grad():
            layer = nn.Linear(2, 3, bias=False)
            x = torch.tensor([1, 2], dtype=torch.float32)

            layer.weight[0,0] = 1
            layer.weight[1,1] = 1
            layer.weight[0,1] = 0
            layer.weight[1,0] = 0
            layer.weight[2,0] = 0
            layer.weight[2,1] = 0

            y = layer(x)
            self.assertTrue(torch.equal(y, torch.tensor([1.0, 2.0, 0.0])))

    def test_nn_linear_empty_batch(self):
        with torch.no_grad():
            layer = nn.Linear(2, 3, bias=False)
            x = torch.tensor([[1, 2]], dtype=torch.float32)

            layer.weight[0,0] = 1
            layer.weight[1,1] = 1
            layer.weight[0,1] = 0
            layer.weight[1,0] = 0
            layer.weight[2,0] = 0
            layer.weight[2,1] = 0

            y = layer(x)
            self.assertTrue(torch.equal(y, torch.tensor([[1.0, 2.0, 0.0]])))
    
    
    def test_nn_linear_batch(self):
        with torch.no_grad():
            layer = nn.Linear(2, 3, bias=False)
            x = torch.tensor([[1, 2],
                              [3, 4],
                              [5, 6]], dtype=torch.float32)

            layer.weight[0,0] = 1
            layer.weight[1,1] = 1
            layer.weight[0,1] = 0
            layer.weight[1,0] = 0
            layer.weight[2,0] = 0
            layer.weight[2,1] = 0

            y = layer(x)
            self.assertTrue(torch.equal(y, torch.tensor([[1.0, 2.0, 0.0],
                                                         [3.0, 4.0, 0.0],
                                                         [5.0, 6.0, 0.0]])))

    def test_nn_linear_meta_batch(self):
        with torch.no_grad():
            layer = nn.Linear(2, 3, bias=False)
            x = torch.tensor([[[1, 2],
                              [3, 4],
                              [5, 6]], 
                              
                             [[1,8],
                              [3, 9],
                              [5, 7]]], dtype=torch.float32)

            layer.weight[0,0] = 1
            layer.weight[1,1] = 1
            layer.weight[0,1] = 0
            layer.weight[1,0] = 0
            layer.weight[2,0] = 0
            layer.weight[2,1] = 0

            y = layer(x)
            self.assertTrue(torch.equal(y, torch.tensor([[[1.0, 2.0, 0.0],
                                                         [3.0, 4.0, 0.0],
                                                         [5.0, 6.0, 0.0]],
                                                         
                                                         [[1.0, 8.0, 0.0],
                                                         [3.0, 9.0, 0.0],
                                                         [5.0, 7.0, 0.0]]])))
            
    def test_serialize_model(self):
        from sklearn.datasets import make_moons

        X_train, y_train = make_moons(n_samples=2, noise=0.1)
        X_validation, y_validation = make_moons(n_samples=2, noise=0.1)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                    torch.tensor(y_train, dtype=torch.long))
        validation_dataset = TensorDataset(torch.tensor(X_validation, dtype=torch.float32),
                                            torch.tensor(y_validation, dtype=torch.long))
        training_loader = DataLoader(train_dataset, shuffle=True)
        validation_loader = DataLoader(validation_dataset)

        device = "cpu"
        in_features = 2
        out_features = 2
        model = nn.Linear(in_features, out_features)
        loss_func = nn.CrossEntropyLoss()
        x = torch.tensor([0.5, 0.4])

        with tempfile.TemporaryDirectory() as tempdir:
            train_config = TrainingConfig(model=model,
                                        loss_func=loss_func,
                                        training_loader=training_loader, 
                                        validation_loader=validation_loader,
                                        save_model=True,
                                        save_path=tempdir,
                                        model_name="simple_model")
            train(train_config)
            y_saved = model(x)
            A_saved = model.weight.data

            model_loaded = nn.Linear(in_features, out_features)

            checkpoint_dict = torch.load(os.path.join(train_config.save_path_final, "last", CHECKPOINT_FILE), map_location=device)
            model_loaded.load_state_dict(checkpoint_dict['model_state_dict'])

            y_loaded = model_loaded(x)
            A_loaded = model_loaded.weight.data
                
        self.assertTrue(torch.equal(A_saved, A_loaded))
        self.assertTrue(torch.equal(y_saved, y_loaded))


class TestMNISTAtenntion(unittest.TestCase):
    def test_torch_flatten(self):
        B, C, H, W = 2, 1, 28, 28
        T = 3 # size of bag
        x = torch.rand([B, T, C, H, W])
        flatten = nn.Flatten()
        x_flatt = flatten(x)
        self.assertTrue(x_flatt.shape == (B, T*C*H*W))

    def test_custom_flatten(self):
        B, C, H, W = 2, 1, 28, 28
        T = 3 # size of bag
        x = torch.rand([B, T, C, H, W])
        flatten = Flatten2()
        x_flatt = flatten(x)
        self.assertTrue(x_flatt.shape == (B, T, C*H*W))

    def test_combiner(self):
        B, C, H, W = 2, 1, 28, 28
        T = 3 # size of bag
        dim_features = 2 # dimension of features vectors h
        x = torch.rand([B, T, C, H, W])
        flatten = Flatten2()
        x_flatt = flatten(x)
        
        featureExtraction = nn.Linear(C*H*W, dim_features)
        weightSelection = nn.Sequential(nn.Linear(dim_features, 1),
                                        nn.Softmax(dim=1))

        features = featureExtraction(x_flatt) 
        weights = weightSelection(features)

        combiner = Combiner(featureExtraction, weightSelection)
        x_final = combiner(x_flatt)
        
        self.assertTrue(x_flatt.shape == (B, T, C*H*W))
        self.assertTrue(features.shape == (B, T, dim_features))
        self.assertTrue(weights.shape == (B, T, 1))
        self.assertTrue(x_final.shape == (B, dim_features))

    def test_attention_dot_score(self):
        B, C, H, W = 2, 1, 28, 28
        T = 3 # size of bag
        dim_features = 2 # dimension of features vectors h
        x = torch.rand([B, T, C, H, W])
        flatten = Flatten2()
        x_flatt = flatten(x)
        self.assertTrue(x_flatt.shape == (B, T, C*H*W))

        featureExtraction = nn.Linear(C*H*W, dim_features)
        weightSelection = DotScore(H)

        features = featureExtraction(x_flatt)
        self.assertTrue(features.shape == (B, T, dim_features))

        features_mean = features.mean(dim=1)
        self.assertTrue(features_mean.shape == (B, dim_features))

        weights = weightSelection(states=features, context=features_mean)        
        self.assertTrue(weights.shape == (B, T, 1))

    @unittest.skip("Foward pass of GeneralScore does not work.")
    def test_attention_general_score(self):
        B, C, H, W = 2, 1, 28, 28
        T = 3 # size of bag
        dim_features = 2 # dimension of features vectors h
        x = torch.rand([B, T, C, H, W])
        flatten = Flatten2()
        x_flatt = flatten(x)
        self.assertTrue(x_flatt.shape == (B, T, C*H*W))

        featureExtraction = nn.Linear(C*H*W, dim_features)
        weightSelection = GeneralScore(H)

        features = featureExtraction(x_flatt)
        self.assertTrue(features.shape == (B, T, dim_features))

        features_mean = features.mean(dim=1)
        self.assertTrue(features_mean.shape == (B, dim_features))

        weights = weightSelection(states=features, context=features_mean)        
        self.assertTrue(weights.shape == (B, T, 1))

    @unittest.skip("Foward pass of AdditiveAttentionScore does not work.")
    def test_attention_additve_score(self):
        B, C, H, W = 2, 1, 28, 28
        T = 3 # size of bag
        dim_features = 2 # dimension of features vectors h
        x = torch.rand([B, T, C, H, W])
        flatten = Flatten2()
        x_flatt = flatten(x)
        self.assertTrue(x_flatt.shape == (B, T, C*H*W))

        featureExtraction = nn.Linear(C*H*W, dim_features)
        weightSelection = AdditiveAttentionScore(H)

        features = featureExtraction(x_flatt)
        self.assertTrue(features.shape == (B, T, dim_features))

        features_mean = features.mean(dim=1)
        self.assertTrue(features_mean.shape == (B, dim_features))

        weights = weightSelection(states=features, context=features_mean)        
        self.assertTrue(weights.shape == (B, T, 1))

    def test_apply_attention(self):
        B, C, H, W = 2, 1, 28, 28
        T = 3 # size of bag
        dim_features = 2 # dimension of features vectors h
        x = torch.rand([B, T, C, H, W])
        flatten = Flatten2()
        x_flatt = flatten(x)
        
        featureExtraction = nn.Linear(C*H*W, dim_features)
        weightSelection = DotScore(H)

        features = featureExtraction(x_flatt) 
        features_mean = features.mean(dim=1)
        scores = weightSelection(features, features_mean)

        combiner = ApplyAttention()
        x_final, weights = combiner(states=features, attention_scores=scores)
        
        self.assertTrue(x_flatt.shape == (B, T, C*H*W))
        self.assertTrue(features.shape == (B, T, dim_features))
        self.assertTrue(scores.shape == (B, T, 1))
        self.assertTrue(x_final.shape == (B, dim_features))
        self.assertTrue(weights.shape == (B, T, 1))

    def test_get_mask(self):
        B, C, H, W = 2, 1, 2, 2
        T = 3 # size of bag
        x = torch.rand([B, T, C*H*W])

        # sequence has only 2 words
        x[0, 2, :] = 0

        # sequence has only 1 words
        x[1, 1, :] = 0
        x[1, 2, :] = 0

        mask_actual = getMaskByFill(x, time_dimension=1, fill=0)
        mask_expected = torch.tensor([[True,
                                       True,
                                       False],
                                      [True,
                                       False,
                                       False]])
      
        self.assertTrue(mask_actual.shape == (B, T))
        self.assertTrue(torch.equal(mask_actual, mask_expected))


    def test_smarter_attention_net(self):
        B, C, H, W = 2, 1, 28, 28
        T = 3 # size of bag
        hidden_size = 2 # dimension of features vectors h
        x = torch.rand([B, T, C, H, W])
        input_size = C * H * W
        out_size = 10


        model = SmarterAttentionNet(input_size, hidden_size, out_size)

        y = model(x)
       
        self.assertTrue(y.shape == (B, out_size))


if __name__ == '__main__':
    unittest.main() 
