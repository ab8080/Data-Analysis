import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_lstm = nn.LSTM(input_size=hid_size, hidden_size=hid_size, batch_first=True)

        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_lstm = nn.LSTM(input_size=hid_size, hidden_size=hid_size, batch_first=True)

        self.category_encoder = nn.Sequential(
            nn.Linear(n_cat_features, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
        )

        self.final_layers = nn.Sequential(
            nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2),
            nn.ReLU(),
            nn.Linear(in_features=hid_size*2, out_features=1)
        )
        
    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1)
        _, (title, _) = self.title_lstm(title_beg)
        title = title[-1]

        full_beg = self.full_emb(input2)
        _, (full, _) = self.full_lstm(full_beg)
        full = full[-1]
        
        category = self.category_encoder(input3)
        
        concatenated = torch.cat([title, full, category], dim=1)
        out = self.final_layers(concatenated)
        
        return out

