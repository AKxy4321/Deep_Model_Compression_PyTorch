import multiprocessing
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_vgg16 import vgg16
from utils import *

# Hyperparameters
batch_size = 128
epochs = 100
learning_rate = 0.001  # Suitable for Adam
# weight_decay = 5e-4  # Commented out as per request
patience = 10  # Early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = multiprocessing.cpu_count()

# Data transformations
transform_val = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(0.8, 1.2)),  # Random zoom-in
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Stronger rotations
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),  # Vary brightness, contrast, etc.
        transforms.ToTensor(),  # Must be before Normalize
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2),  # Randomly erase part of image
    ]
)

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root=dataset_path, train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root=dataset_path, train=False, download=True, transform=transform_val
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

# Initialize model, loss, and optimizer
model = vgg16().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Changed to Adam


def train_and_evaluate():
    best_accuracy = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(os.getcwd(), "weights", "vgg16.pt")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            train_loader_tqdm.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )

        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print("Model checkpoint saved with best validation accuracy!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break


train_and_evaluate()
print("Training complete!")
