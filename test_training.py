#!/usr/bin/env python3
"""Quick test of the training pipeline."""

import os
import torch
import torchvision.models as models
from PIL import Image
from pathlib import Path

print("="*60)
print("Training Pipeline Test")
print("="*60)

# Test 1: Check dataset
print("\n[1/4] Checking dataset...")
data_dir = 'data'
if not os.path.exists(data_dir):
    print(f"[ERROR] Data directory not found: {data_dir}")
    exit(1)

classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
total_images = 0
for cls in classes:
    cls_dir = os.path.join(data_dir, cls)
    if os.path.exists(cls_dir):
        count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {cls:20} : {count:4} images")
        total_images += count
    else:
        print(f"  {cls:20} : NOT FOUND")

print(f"  Total images: {total_images}")

if total_images == 0:
    print("[ERROR] No images found!")
    exit(1)

print("[OK] Dataset check passed")

# Test 2: Load sample image
print("\n[2/4] Testing image loading...")
try:
    sample_class = os.path.join(data_dir, classes[0])
    sample_image = os.path.join(sample_class, os.listdir(sample_class)[0])
    img = Image.open(sample_image).convert('RGB')
    img = img.resize((224, 224))
    print(f"  Loaded: {sample_image}")
    print(f"  Size: {img.size}")
    print("[OK] Image loading passed")
except Exception as e:
    print(f"[ERROR] Error loading image: {e}")
    exit(1)

# Test 3: Load PyTorch model
print("\n[3/4] Testing MobileNetV2 model...")
try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    print(f"  Model loaded successfully")
    print(f"  Last layer: {model.classifier[1]}")
    print("[OK] Model loading passed")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    exit(1)

# Test 4: Quick inference
print("\n[4/4] Testing inference...")
try:
    import torch
    import torchvision.transforms as transforms

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0)
    print(f"  Input tensor shape: {img_tensor.shape}")

    with torch.no_grad():
        output = model(img_tensor)

    print(f"  Output shape: {output.shape}")
    print("[OK] Inference passed")

except Exception as e:
    print(f"[ERROR] Error during inference: {e}")
    exit(1)

print("\n" + "="*60)
print("[OK] All tests passed! Ready to train.")
print("="*60)
print("\nTo start training, run:")
print("  python train.py")
