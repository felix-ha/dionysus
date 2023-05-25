
import unittest
import torch
import torch.nn as nn
from image import *


class TestImage(unittest.TestCase):
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




        
