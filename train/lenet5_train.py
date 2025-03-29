import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add parent directory to sys.path

import torch

# Define the LeNet model
import torch.nn as nn
import torch.optim as optim

from train.train_config import device, train_config


def LeNet5():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, bias=True),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, bias=True),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=50 * 4 * 4, out_features=500),
        nn.ReLU(),
        nn.Linear(in_features=500, out_features=10),
    )


# Hyperparameters
batch_size = 128
train_loader, test_loader = train_config(batch_size=batch_size, dataset=0)
epochs = 100
learning_rate = 0.001
patience = 10

# Initialize model, loss, and optimizer
model = LeNet5().to(device)
# model.load_state_dict(
#     torch.load(os.path.join(os.getcwd(), "weights", "lenet5.pt"), weights_only=True)
# )
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop with early stopping and model checkpoint
def train_and_evaluate():
    best_accuracy = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(os.getcwd(), "weights", "lenet5.pt")

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
