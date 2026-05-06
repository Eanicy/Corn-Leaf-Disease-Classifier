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
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Class labels
CLASS_LABELS = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
CLASS_MAPPING = {label: idx for idx, label in enumerate(CLASS_LABELS)}

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class CornLeafDataset(Dataset):
    """Custom dataset for corn leaf images."""
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


def load_and_preprocess_data(data_dir):
    """Load images from folder structure."""
    image_paths = []
    labels = []

    for class_label in CLASS_LABELS:
        class_dir = os.path.join(data_dir, class_label)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue

        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                img_path = os.path.join(class_dir, img_file)
                try:
                    Image.open(img_path)
                    image_paths.append(img_path)
                    labels.append(CLASS_MAPPING[class_label])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

    return np.array(image_paths), np.array(labels)


def compute_class_weights(labels):
    """Compute class weights to handle imbalance."""
    label_counts = Counter(labels)
    total = len(labels)
    weights = {}

    for class_idx, count in label_counts.items():
        weights[class_idx] = total / (len(label_counts) * count)

    return weights


def split_data(image_paths, labels):
    """Split into train/val/test with stratification."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_data_loaders(train_data, val_data, test_data):
    """Create PyTorch DataLoaders with aggressive augmentation for real-world robustness."""
    train_transform = transforms.Compose([
        # Aggressive rotation and perspective transforms
        transforms.RandomRotation(60),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.3, 0.3),  # More aggressive translation
            scale=(0.5, 1.5),       # Stronger zoom/crop
            shear=(-20, 20)         # Add shear distortion
        ),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # Realistic camera angles

        # Aggressive flip and rotation variants
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),

        # Extreme lighting and color variations (phone camera conditions)
        transforms.ColorJitter(
            brightness=(0.3, 1.7),   # Extreme brightness shifts
            contrast=(0.4, 2.0),      # Strong contrast variations
            saturation=(0.4, 1.8),    # Saturation shifts
            hue=(-0.2, 0.2)           # Color cast variations
        ),

        # Blur variations (focus issues, motion blur)
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),

        # Random crops with different aspect ratios (simulating different framing)
        transforms.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=(0.4, 1.0),  # Allow significant crop variation
            ratio=(0.75, 1.33) # Allow aspect ratio changes
        ),

        # Convert to tensor
        transforms.ToTensor(),

        # Normalize using ImageNet statistics
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_dataset = CornLeafDataset(X_train, y_train, transforms=train_transform)
    val_dataset = CornLeafDataset(X_val, y_val, transforms=val_transform)
    test_dataset = CornLeafDataset(X_test, y_test, transforms=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def build_model(num_classes):
    """Build MobileNetV2 transfer learning model."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)

    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=EPOCHS):
    """Train the model."""
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    return model


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
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

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = (all_predictions == all_labels).mean()

    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"Test Accuracy: {accuracy:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )

    print("\nPer-Class Metrics:")
    for idx, label in enumerate(CLASS_LABELS):
        print(f"{label:20} | Precision: {precision[idx]:.4f} | Recall: {recall[idx]:.4f} | F1: {f1[idx]:.4f}")

    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)

    return {'accuracy': float(accuracy)}


def export_to_onnx(model, export_dir='models'):
    """Export PyTorch model to ONNX format."""
    os.makedirs(export_dir, exist_ok=True)

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    onnx_path = os.path.join(export_dir, 'corn_disease_classifier.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=False
    )

    print(f"\nModel exported to {onnx_path}")

    # Export label mapping
    labels_path = os.path.join(export_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump({
            'labels': CLASS_LABELS,
            'mapping': CLASS_MAPPING
        }, f, indent=2)

    print(f"Labels exported to {labels_path}")

    # Save PyTorch model
    pt_path = os.path.join(export_dir, 'corn_disease_classifier.pt')
    torch.save(model.state_dict(), pt_path)
    print(f"PyTorch model exported to {pt_path}")

    return onnx_path


def main():
    print("Loading dataset...")
    data_dir = 'data'
    image_paths, labels = load_and_preprocess_data(data_dir)

    print(f"Loaded {len(image_paths)} images")
    print(f"Class distribution: {dict(Counter(labels))}")

    # Split data
    print("\nSplitting data (70/15/15)...")
    train_data, val_data, test_data = split_data(image_paths, labels)
    print(f"Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(train_data, val_data, test_data)

    # Build model
    print("\nBuilding MobileNetV2 model...")
    model = build_model(num_classes=len(CLASS_LABELS)).to(DEVICE)

    # Loss and optimizer
    class_weights = compute_class_weights(train_data[1])
    weights_tensor = torch.tensor([class_weights[i] for i in range(len(CLASS_LABELS))], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Train model
    print(f"\nTraining model on {DEVICE}...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=EPOCHS)

    # Evaluate
    print("\nEvaluating model...")
    evaluate_model(model, test_loader)

    # Export
    print("\nExporting model...")
    export_to_onnx(model)

    print("\n[OK] Training complete!")


if __name__ == '__main__':
    main()
