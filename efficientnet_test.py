import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from utils.datasets import CustomImageDataset


def run_experiment():
    TRAIN_DIR = os.path.join('data', 'train')
    TEST_DIR = os.path.join('data', 'test')
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = CustomImageDataset(root_dir=TRAIN_DIR, transform=transform, target_size=(224, 224))
    val_ds = CustomImageDataset(root_dir=TEST_DIR,   transform=transform, target_size=(224, 224))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(train_ds.get_class_names())

    effnet_weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=effnet_weights)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = running_correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss/total
        train_acc = running_correct/total

        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{NUM_EPOCHS} — "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    epochs = list(range(1, NUM_EPOCHS+1))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], marker='o', label='Train')
    plt.plot(epochs, history['val_loss'],   marker='o', label='Val')
    plt.title('Loss vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], marker='o', label='Train')
    plt.plot(epochs, history['val_acc'],   marker='o', label='Val')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/efficientnet/learning_curve.png")


if __name__ == '__main__':
    run_experiment()
