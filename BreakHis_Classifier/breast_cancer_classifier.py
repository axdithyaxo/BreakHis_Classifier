import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from sklearn.metrics import f1_score

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Data directories
    train_dir = "data/train"
    val_dir = "data/val"

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets and loaders
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.head = nn.Linear(model.head.in_features, 2)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            train_bar.set_postfix(loss=loss.item())

        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        train_epoch_f1 = f1_score(train_labels, train_preds, average='weighted')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                val_bar.set_postfix(loss=loss.item())

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_epoch_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_epoch_loss:.4f} Train Acc: {train_epoch_acc:.4f} Train F1: {train_epoch_f1:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f} Val F1: {val_epoch_f1:.4f}")

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/breast_vit_classifier_epoch{epoch+1}.pth")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()