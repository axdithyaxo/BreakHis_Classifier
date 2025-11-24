import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import os

# Device configuration: MPS > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define the model class (assuming the model is a ViT-based classifier)
# If the model class is defined elsewhere, import it accordingly
import timm
import torch.nn as nn

# Load the saved model checkpoint
checkpoint_path = "models/breast_vit_classifier_epoch8.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Rebuild the same ViT architecture and load weights
model = timm.create_model("vit_base_patch16_224", pretrained=False)
model.head = nn.Linear(model.head.in_features, 2)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
# Define transforms for validation data
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load validation dataset
val_dir = "data/val"
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluate the model
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
cm = confusion_matrix(all_labels, all_preds)

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation F1 Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_dataset.classes,
            yticklabels=val_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Display 4 random predictions from the validation set
indices = np.random.choice(len(val_dataset), 4, replace=False)
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, idx in enumerate(indices):
    image, true_label = val_dataset[idx]
    input_img = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_img)
        _, pred_label = torch.max(output, 1)
    pred_label = pred_label.item()
    true_label_name = val_dataset.classes[true_label]
    pred_label_name = val_dataset.classes[pred_label]
    image_np = image.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    axs[i].imshow(image_np)
    axs[i].axis('off')
    axs[i].set_title(f"Pred: {pred_label_name}\nTrue: {true_label_name}")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import timm

# Select one image to visualize
img_path = "data/val/malignant/SOB_M_DC-14-2523-40-016.png" # change to any sample image
img = Image.open(img_path).convert("RGB")

# Same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# --- Evaluate on 100x images ---
test_dir = "data/archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549CD/100X/"
test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"\n🔍 100X Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")