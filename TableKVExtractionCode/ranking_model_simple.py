import numpy as np
import torch
import torch.nn as nn
import sklearn as sk


class ranking_model(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ranking_model, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.cosine_similarity = nn.CosineSimilarity(1, 1e-6)

    def forward(self, vector, dictionary):
        encoded_vector = self.vector_encode(torch.transpose(vector, 0, 1))
        dictionary_vector = self.vector_encode(torch.transpose(dictionary, 0, 1))
        cosine_distance = self.cosine_similarity(encoded_vector, dictionary_vector)
        return cosine_distance

    def vector_encode(self, vector):
        return self.linear(vector)
