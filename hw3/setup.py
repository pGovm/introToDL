# Importing the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

import time
import sys
import os
import random
import numpy as np
import subprocess
import matplotlib.pyplot as plt


try:
    import nltk
except ImportError:
    print("NLTK library not found. Intalling now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Some common variables are initialised here
SOS_token = 0
EOS_token = 1
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting up the base path regardless of device
from pathlib import Path
BASE_DIR = Path(__file__).parent

# Loading the file into an ordered list
def load(filePath):
    pairs = []

    with open(filePath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line:
                eng, fra = line.split('\t')
                pairs.append((eng,fra))
    
    return pairs

dataset = load(BASE_DIR/ "data/vast_english_french.txt")

# Preparing the required mappings
unique_chars = sorted(list(set(''.join([word for pair in dataset for word in pair]))))
chars_to_idx = {'SOS': SOS_token, 'EOS': EOS_token}

for i, char in enumerate(unique_chars):
    chars_to_idx[char] = i + 2

idx_to_chars = {i: char for char, i in chars_to_idx.items()}

# Converting the list into a pytorch dataset
class TranslationDataset(Dataset):
    def __init__(self, dataset, char_to_idx):
        self.dataset = dataset
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_wrd, target_wrd = self.dataset[idx]
        inp_tensor = torch.tensor([self.char_to_idx[char] for char in input_wrd] + [EOS_token], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_idx[char] for char in target_wrd] + [EOS_token], dtype=torch.long)

        return inp_tensor, target_tensor
    
dataset_torch = TranslationDataset(dataset, chars_to_idx)

# Splitting the data
train_size = int(0.8 * len(dataset_torch))
val_size = len(dataset_torch) - train_size

train_ds, val_ds = random_split(dataset_torch, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

# Loading the dataset
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

input_size = len(chars_to_idx)
hidden_size = 128
output_size = len(chars_to_idx)
no_of_epochs = 50
max_length = 100

# Graphing helper functions
def plot_loss(train_losses: list, val_losses: list, title: str, path: str):
    
    plt.figure(figsize=(10, 5))
    
    # Plotting figure
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    # Applying labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title}')

    plt.legend()
    plt.tight_layout()
   
    # Saving figure
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")    
    
    plt.show()