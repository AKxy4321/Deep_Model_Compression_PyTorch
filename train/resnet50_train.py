import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add parent directory to sys.path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.model_resnet50 import resnet50
from train.train_config import device, train_config

# Hyperparameters
batch_size = 128
train_loader, test_loader = train_config(batch_size=batch_size, dataset=1)
epochs = 100
learning_rate = 0.001  # Suitable for Adam
# weight_decay = 5e-4  # Commented out as per request
patience = 10  # Early stopping patience

# Initialize model, loss, and optimizer
model = resnet50(pretrained=False).to(
    device  # make this True when you want to train with weights
)
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
