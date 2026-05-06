# CornDoctor 🌽🩺

**Diagnose corn leaf diseases instantly with your phone camera**

CornDoctor is a mobile app that uses artificial intelligence to identify corn plant diseases in real-time. Simply point your phone camera at a corn leaf, and the app will tell you if it's healthy or what disease it has.

---

## ✨ What Can It Do?

📸 **Take photos** — Use your phone camera to capture corn leaves  
🤖 **Instant diagnosis** — AI analyzes the image in seconds  
🎯 **4 disease types** — Identifies Blight, Rust, Gray Leaf Spot, or Healthy leaves  
💬 **Get advice** — Chat with the AI about disease treatment  
📱 **Works offline** — No internet connection needed  
⚡ **Fast** — Results in under 1 second  

---

## 🌾 The Data

We trained the AI on **4,188 real corn leaf photos** from farmers:

- **1,162 healthy leaves** (27.7%)
- **1,306 Common Rust cases** (31.2%)
- **1,146 Blight cases** (27.4%)
- **574 Gray Leaf Spot cases** (13.7%)

---

## 📊 How Accurate Is It?

**Test Results: 93.64% Accurate**

When we tested it on new photos it had never seen before:
- Correctly identified **99.4% of healthy leaves**
- Correctly identified **97.9% of Common Rust**
- Correctly identified **98.6% of healthy plants**
- Correctly identified **84.5% of Gray Leaf Spot**

**Real-World Usage:** Works well on actual corn plant photos. If you photograph a monitor screen or non-corn image, it will ask you to take a real corn photo instead.

---

## 📱 How to Use the App

### Step 1: Open the App
Launch CornDoctor on your Android phone

### Step 2: Take a Photo
- Tap **"Take Photo"** to use your camera, or
- Tap **"Choose from Gallery"** to pick an existing photo

### Step 3: Point at a Corn Leaf
- Make sure the leaf is clear and well-lit
- Avoid shadows or blurry photos
- Focus on the diseased area (if any)

### Step 4: See Results
The app instantly shows:
- ✅ Disease name (or "Healthy")
- 📊 Confidence percentage (how sure the AI is)
- 📖 Disease information (what to do about it)

### Step 5: Get Advice
Tap **"Ask Chatbot"** to chat with the AI about:
- How to treat the disease
- What products to use
- Disease prevention tips

---

## 🚀 Training Your Own Model

### What You Need
- A computer with **Python 3** installed
- About **2-4 hours** for training (or 30 minutes on a gaming PC)
- Your own corn leaf photos (organized by disease type)

### Step 1: Organize Your Photos

Create folders like this:
```
my_corn_data/
├── Blight/          (put all blight photos here)
├── Rust/            (put all rust photos here)
├── Gray_Leaf_Spot/  (put all gray leaf spot photos here)
└── Healthy/         (put all healthy leaf photos here)
```

### Step 2: Download the Training Code

Get the `train.py` file from this project

### Step 3: Install Requirements

Open a terminal/command prompt and type:
```
pip install -r requirements.txt
```

This downloads the AI tools needed.

### Step 4: Start Training

In the terminal, type:
```
python train.py
```

The computer will:
1. Load your photos
2. Learn patterns from them
3. Create a trained AI model
4. Save it as `corn_disease_classifier.pt`

That's it! Your custom model is ready.

### Step 5: Convert for Mobile (Optional)

If you want to use your trained model in the app:

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy-paste this code:
```python
!pip install onnx2tf tensorflow
import subprocess
subprocess.run(['onnx2tf', '-i', 'corn_disease_classifier.onnx', '-o', 'converted_model'])
```
4. Download the `.tflite` file
5. Place it in the app's model folder

---

## 📂 Project Files

```
corndoctor/
├── train.py                    ← Run this to train the model
├── requirements.txt            ← List of tools needed
├── data/                       ← Your corn photos go here
├── models/                     ← Trained AI models saved here
└── flutter_app/                ← The mobile app
    └── assets/models/          ← AI model for the app
```

---

## ⚠️ Important Notes

### What Works Well
✅ Photos of actual corn plant leaves  
✅ Good lighting (natural sunlight best)  
✅ Clear, focused images  
✅ Direct camera shots (not through windows/screens)  

### What Doesn't Work
❌ Photos of corn pictures on a monitor  
❌ Drawings or artwork of corn leaves  
❌ Blurry or dark photos  
❌ Images of other plants mistaken for corn  

**If the app says "Please capture a corn leaf"** — it means:
- The image is too dark
- The leaf isn't clear enough
- It's not actually a corn leaf
- Try retaking the photo with better lighting

---

## 🔮 Coming Soon

🚀 **Multi-model choice** — Compare different AI models  
🎯 **Object detection** — Shows exactly WHERE the disease is  
💬 **Better chatbot** — More natural conversations  
🍎 **iOS support** — Works on iPhones too  

---

## ❓ Troubleshooting

### "My app crashes on startup"
- Update Flutter: `flutter upgrade`
- Rebuild: `flutter clean` then `flutter run`

### "Model accuracy is low on my photos"
- Make sure photos show clear leaf damage
- Use good lighting (avoid shadows)
- Take multiple photos from different angles

### "Training takes too long"
- If using CPU: This is normal, takes 2-4 hours
- If you have NVIDIA GPU: Update to use GPU acceleration

### "I want better accuracy"
- Collect more photos of diseases you see often
- Retrain with your own data
- Photograph actual plants, not pictures

---

## 💡 Tips for Best Results

1. **Good Lighting** — Use natural sunlight
2. **Clear Focus** — Make sure the leaf is sharp, not blurry
3. **Show the Problem** — Zoom in on the diseased area
4. **Take Multiple Photos** — Different angles give better results
5. **Real Plants** — Use actual corn leaves, not pictures

---

## 📝 What Disease Am I Looking At?

**Healthy Leaf**
- Bright green color
- No spots or marks
- Smooth surface

**Common Rust**
- Small brown/reddish bumps
- Scattered across the leaf
- Looks like a dusty coating

**Blight (Northern Corn Leaf Blight)**
- Long, narrow gray-tan streaks
- Runs along leaf veins
- Gets larger over time

**Gray Leaf Spot**
- Rectangular gray-brown spots
- Dark borders/edges
- Starts small, spreads

---

## 🎓 How Does It Work?

The app uses a technique called **transfer learning**:

1. We start with an AI model trained on millions of images (ImageNet)
2. We teach it specifically about corn diseases using our 4,188 photos
3. The AI learns to recognize disease patterns
4. When you take a photo, the AI compares it to what it learned
5. It gives you a confidence score (how sure it is)

**Key Feature:** If confidence is below 60%, the app rejects it and asks for a better photo. This prevents wrong guesses.

---

## 🤝 Help Us Improve

Have suggestions? Found a bug? Want to add features?

**You can help by:**
- 📸 Sharing your corn photos to improve the model
- 🐛 Reporting bugs you find
- 💡 Suggesting new features
- 📢 Spreading the word

---

## 📄 License

This project is open source and free to use.

---

## 📧 Questions?

Email: [your.email@example.com]

---

**Made with ❤️ for farmers and corn growers**

🌽 CornDoctor — Healthier crops, better harvests 🌽
