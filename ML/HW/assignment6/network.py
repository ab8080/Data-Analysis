import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class TextProcessor(nn.Module):
    def __init__(self, n_tokens, hid_size):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=hid_size, out_channels=hid_size, kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten()
        )
    
    def forward(self, x):
        x = self.embedding(x).permute((0, 2, 1))
        return self.encoder(x)

class CategoryProcessor(nn.Module):
    def __init__(self, n_cat_features):
        super(CategoryProcessor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_cat_features, 600),
            nn.ReLU(),
            nn.Linear(600, 100),
            nn.ReLU(),
            nn.Linear(100, 20)
        )
        
    def forward(self, x):
        return self.layers(x)

class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_processor = TextProcessor(n_tokens, hid_size)
        self.full_processor = TextProcessor(n_tokens, hid_size)
        self.category_processor = CategoryProcessor(n_cat_features)
        self.final_layers = nn.Sequential(
            nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2),
            nn.ReLU(),
            nn.Linear(in_features=hid_size*2, out_features=1)
        )

    def forward(self, whole_input):
        title, full, category = whole_input
        title = self.title_processor(title)
        full = self.full_processor(full)
        category = self.category_processor(category)

        concatenated = torch.cat([title, full, category], dim=1)
        return self.final_layers(concatenated)

