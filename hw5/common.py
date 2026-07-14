# Importing the required libraries
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from torchinfo import summary
from tqdm import tqdm

# Initializing the device we will be using
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cifar100(batch_size, image_size=32, mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)):
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Initializing the required dataset
    train_ds = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

    # Initializing the loaders
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_flops(model, input_size):
    stats = summary(model=model, input_size=input_size, verbose=0)

    return stats.total_mult_adds

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
        
        accuracy = (correct / total) * 100

        return accuracy
    
def train_model(model, train_loader, test_loader, lr, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Adding scheduling to improve test acc
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    epoch_time = 0.0

    for epoch in range(epochs):
        start = time.time()

        # Training loop
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f'Epoch ({epoch+1}/{epochs})')

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs # Check in-place for HuggingFace return type
            
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()

            # Adding clipping to try to improve test accuracy
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
        
        scheduler.step()

        avg_loss = train_loss / len(train_loader)
        train_time = time.time() - start
        epoch_time += train_time

        print(f'Epoch: {epoch+1}    |    Loss: {avg_loss:.4f}    |    Time: {train_time:.3f} sec')

    avg_epoch_time = epoch_time / epochs
    test_acc = evaluate(model=model, test_loader=test_loader, device=device)
    num_params = count_params(model=model)

    return {
        "test_acc": test_acc,
        "avg_epoch_time": avg_epoch_time,
        "num_params": num_params
    }   