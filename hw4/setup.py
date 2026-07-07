# Importing all required libraries
import os
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from thop import profile, clever_format

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from pathlib import Path

## Setting up the base path
BASE_DIR = Path(__file__).parent

# Common variables to use
SOS_token = 0
EOS_token = 1
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer configuration list
transformer_configs = [
    {'num_layers': 1, 'nhead': 2},
    {'num_layers': 1, 'nhead': 4},
    {'num_layers': 2, 'nhead': 2},
    {'num_layers': 2, 'nhead': 4},
    {'num_layers': 4, 'nhead': 2},
    {'num_layers': 4, 'nhead': 4},
]



################################################
############    Class Definitions   ############
################################################

# Defining classes to split and prepare data
class CharDataset(Dataset):
    def __init__(self, encoded_text, seq_len):
        self.encoded_text = encoded_text
        self.seq_len = seq_len

    def __len__(self):
        return len(self.encoded_text) - self.seq_len - 1
    
    def __getitem__(self, index):
        sequence = self.encoded_text[index: (index + self.seq_len)]
        target = self.encoded_text[(index + 1): (index + self.seq_len + 1)]

        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)

        return sequence_tensor, target_tensor

class TranslationDataset(Dataset):
    def __init__(self, dataset, chars_to_idx):
        self.dataset = dataset
        self.chars_to_idx = chars_to_idx

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        input, target = self.dataset[index]
        input_tensor = torch.tensor([self.chars_to_idx[char] for char in input] + [EOS_token], dtype=torch.long)
        target_tensor = torch.tensor([self.chars_to_idx[char] for char in target] + [EOS_token], dtype=torch.long)

        return input_tensor, target_tensor
    
# Creating the class that creates positional encoding for the transformer
class positionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(positionalEncoding, self).__init__()
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))

        self.encoding = torch.zeros(max_len, d_model)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()
    
# Transformer encoder class for problems 1 and 2
class charTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(charTransformer, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = positionalEncoding(hidden_size)
        
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output)

        return output
    
# Transformer encoder-decoder class for problems 3 and 4
class translationTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(translationTransformer, self).__init__()

        # Encoding layer
        self.encoder_embedding = nn.Embedding(input_size, hidden_size)
        self.encoder_pos_encoder = positionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)

        # Decoding layer
        self.decoder_embedding = nn.Embedding(output_size, hidden_size)
        self.decoder_pos_encoder = positionalEncoding(hidden_size)
        decoder_layers = nn.TransformerDecoderLayer(hidden_size, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, target):
        # Encoding source
        encoder_embedded = self.encoder_embedding(input)
        encoder_embedded = self.encoder_pos_encoder(encoder_embedded)
        encoder_output = self.transformer_encoder(encoder_embedded)

        # Decoding target
        decoder_embedded = self.decoder_embedding(target)
        decoder_embedded = self.decoder_pos_encoder(decoder_embedded)
        decoder_output = self.transformer_decoder(decoder_embedded, encoder_output)

        # Transformer output
        output = self.fc(decoder_output)

        return output

###############################################
############    Helper Functions   ############
###############################################

# Function to load the required files:
def load_file(translation, filePath):
    pairs = []

    with open(filePath, encoding='utf-8') as f:
        if translation:
            for line in f:
                line = line.strip()

                if line:
                    eng, fra = line.split('\t')
                    pairs.append((eng, fra))
            return pairs
        else:
            return f.read()
        
# Function to create the required mappings for each training        
def vocab(translation, dataset):
    if translation:
        unique_chars = sorted(list(set(''.join([word for pair in dataset for word in pair]))))
        chars_to_idx = {'SOS':SOS_token, 'EOS':EOS_token}

        for i, char in enumerate(unique_chars):
            chars_to_idx[char] = i + 2
        
        idx_to_chars = {i: ch for ch, i in chars_to_idx.items()}

        return chars_to_idx, idx_to_chars, None
    else:
        chars = sorted(list(set(dataset)))

        chars_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_chars = {i: ch for i, ch in enumerate(chars)}
        encoded_text = [chars_to_idx[ch] for ch in dataset]

        return chars_to_idx, idx_to_chars, encoded_text
    
def build_loaders(dataset, batch_size):
    # Defining train/val sizes
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Splitting the dataset
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    # Creating data loaders for train/val
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)

    return train_loader, val_loader

# Function to compute the size of the trained models
def compute_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024**2)

    return size_mb

# Function to compute time taken to train models
def compute_time(start, end):
    return end - start

# Helper function for plotting losses
def plot_loss(train_losses, val_losses, title, path):
    plt.figure(figsize=(10,5))

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Model Loss')
    plt.title(f'{title}')
    plt.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")

    plt.show()

# Helper function to calculate computational complexity
def compute_flops(model, *inputs):
    flops, params, *_ = profile(model, inputs=inputs, verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    return flops, params

