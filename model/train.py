import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time

from model import EdgeCNN
from data import get_dataloaders



def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one full training epoch and returns average loss and accuracy
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc



@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation/test set
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Metrics
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc




def main():
    # Hyperparameters
    num_epochs = 30
    learning_rate = 4e-3
    label_smoothing = 0.1

    
    print("=" * 50)
    print("Training EdgeCNN".center(50))
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(batch_size=128)

    # Model
    model = EdgeCNN(num_classes=10).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-2
    )

    # Learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs
    )


    # ========================= #
    # Training loop
    # ========================= #
    best_val_acc = 0.0
    train_start = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model,test_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(f"{'Train':<6} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"{'Val':<6} | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"Epoch Time: {epoch_time:.4f}s")


        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "edgecnn_v1.2.pth")

    train_time = time.time() - train_start

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Total training time: {train_time:.4f}s")


if __name__ == "__main__":
    main()
