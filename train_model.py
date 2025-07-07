import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import DataLoader
from datasets import CustomImageDataset

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "homework_5/results/"

def load_datasets():
    """Загружает тренировочный и валидационный датасеты"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = CustomImageDataset("homework_5/data/train", transform=transform)
    val_dataset = CustomImageDataset("homework_5/data/val", transform=transform)

    return train_dataset, val_dataset

def prepare_model(num_classes):
    """Загружает EfficientNet и заменяет классификатор"""
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model.to(DEVICE)

def train_one_epoch(model, loader, optimizer, criterion):
    """Тренирует модель на одной эпохе"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def validate(model, loader, criterion):
    """Оценивает модель на валидационном наборе"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer):
    """Выполняет полный цикл обучения"""
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies

def save_model(model):
    """Сохраняет обученную модель"""
    path = os.path.join(RESULTS_DIR, "efficientnet_b0_finetuned.pth")
    torch.save(model.state_dict(), path)
    print(f"Модель сохранена: {path}")

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Строит и сохраняет графики потерь и точности"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss по эпохам")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy по эпохам")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "training_plot.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Графики сохранены в {plot_path}")

def main():
    train_dataset, val_dataset = load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = prepare_model(num_classes=len(train_dataset.get_class_names()))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer
    )

    save_model(model)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

if __name__ == "__main__":
    main()