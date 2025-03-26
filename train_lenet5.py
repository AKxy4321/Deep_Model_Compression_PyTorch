import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.optim as optim
import multiprocessing
import torch.nn as nn
from utils import *
import torch
import os


# Define the LeNet model
def LeNet5():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=2, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=2, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=50 * 4 * 4, out_features=500),
        nn.ReLU(),
        nn.Linear(in_features=500, out_features=10),
        nn.Softmax(dim=1),
    )


# Hyperparameters
batch_size = 128
epochs = 100
learning_rate = 0.001
patience = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = multiprocessing.cpu_count()

# Data transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Load MNIST dataset
train_dataset = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=dataset_path, train=False, download=True, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

# Initialize model, loss, and optimizer
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop with early stopping and model checkpoint
def train_and_evaluate():
    best_accuracy = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(os.getcwd(), "models", "lenet_best.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
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

        # Save best model based on validation accuracy
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


# Run training and evaluation loop
train_and_evaluate()
print("Training complete!")
