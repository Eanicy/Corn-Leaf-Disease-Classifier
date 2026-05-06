import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import Counter
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # Quick test with 5 epochs
LEARNING_RATE = 0.001
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Class labels
CLASS_LABELS = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
CLASS_MAPPING = {label: idx for idx, label in enumerate(CLASS_LABELS)}

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")


class CornLeafDataset(Dataset):
    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, label


def load_data(data_dir):
    print(f"[INFO] Loading images from {data_dir}...")
    image_paths = []
    labels = []

    for class_label in CLASS_LABELS:
        class_dir = os.path.join(data_dir, class_label)
        if not os.path.exists(class_dir):
            print(f"[WARN] {class_dir} not found")
            continue

        files = [f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        for img_file in files[:100]:  # Limit to 100 images per class for quick test
            img_path = os.path.join(class_dir, img_file)
            try:
                Image.open(img_path)
                image_paths.append(img_path)
                labels.append(CLASS_MAPPING[class_label])
            except:
                pass

    return np.array(image_paths), np.array(labels)


def get_loaders(train_data, val_data, test_data):
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CornLeafDataset(train_data[0], train_data[1], transforms=train_transform)
    val_dataset = CornLeafDataset(val_data[0], val_data[1], transforms=val_transform)
    test_dataset = CornLeafDataset(test_data[0], test_data[1], transforms=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


def main():
    # Load data
    image_paths, labels = load_data('data')
    print(f"[INFO] Loaded {len(image_paths)} images")
    print(f"[INFO] Class distribution: {dict(Counter(labels))}")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create loaders
    train_loader, val_loader, test_loader = get_loaders(
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    )

    # Build model
    print("[INFO] Building MobileNetV2...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2).to(DEVICE)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_LABELS))
    model = model.to(DEVICE)

    # Loss and optimizer
    class_weights = {i: len(y_train) / (len(CLASS_LABELS) * (y_train == i).sum())
                     for i in range(len(CLASS_LABELS))}
    weights_tensor = torch.tensor([class_weights[i] for i in range(len(CLASS_LABELS))],
                                  dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Train
    print("[INFO] Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_loss)
        print(f"[EPOCH {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Evaluate
    print("\n[INFO] Evaluating...")
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    print(f"\n[RESULT] Test Accuracy: {accuracy:.4f}")

    # Export model
    print("\n[INFO] Exporting model...")
    os.makedirs('models', exist_ok=True)

    # Save PyTorch model
    torch.save(model.state_dict(), 'models/corn_disease_classifier.pt')
    print("[OK] PyTorch model saved to models/corn_disease_classifier.pt")

    # Save ONNX
    try:
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        torch.onnx.export(model, dummy_input, 'models/corn_disease_classifier.onnx',
                         export_params=True, opset_version=13, do_constant_folding=True)
        print("[OK] ONNX model saved to models/corn_disease_classifier.onnx")
    except Exception as e:
        print(f"[WARN] Could not export ONNX: {e}")

    # Save labels
    with open('models/labels.json', 'w') as f:
        json.dump({'labels': CLASS_LABELS, 'mapping': CLASS_MAPPING}, f, indent=2)
    print("[OK] Labels saved to models/labels.json")

    print("\n[OK] Quick training completed successfully!")
    print("[INFO] Run 'python train.py' for full training with all images and 30 epochs")


if __name__ == '__main__':
    main()
