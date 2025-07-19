# 🧠 Parkinson Disease Detection Using Multimodal Deep Learning

This project focuses on the multimodal classification of Parkinson’s Disease by analyzing both **image frames** and **audio features** extracted from YouTube interviews. Two deep learning models were developed using Early Fusion and Late Fusion strategies to detect Parkinson’s symptoms from both modalities.

---

## 📌 Project Summary

Parkinson’s Disease is a progressive neurological disorder that affects movement. Early diagnosis is crucial for slowing disease progression and improving quality of life.

This project uses multimodal data — **audio** (speech) and **visual** (facial movements) — to train deep learning models that can detect Parkinson’s Disease automatically.

---

## 🧰 Technologies Used

- Python
- PyTorch
- Librosa (Audio feature extraction - MFCC)
- OpenCV (Image preprocessing)
- MoviePy, Pytube (YouTube scraping)
- Pandas, NumPy
- Audacity (Audio cleaning)
- Scikit-learn (Metrics & Evaluation)
- Matplotlib (Graphs)

---

## 📁 Project Pipeline

### 1. Data Collection

- **12 YouTube videos** (interviews of both Parkinson’s patients and healthy individuals) were collected using `pytube`.
- **Audio tracks** were extracted with `moviepy`.
- **image frames** were extracted with `OpenCV`.

### 2. Audio Preprocessing

- Silence and background noise were removed using Audacity.
- Audio was split into **5-second segments**.
- **MFCC (Mel-Frequency Cepstral Coefficients)** were extracted using `librosa`.

### 3. Image Preprocessing

- Frames were  cleaned and cropped to include only facial regions.
- Transforms (resizing, flipping, rotation, etc.) were applied for data augmentation.

### 4. Dataset Construction

- Each sample includes:  
  - A cleaned frame  
  - Corresponding MFCC feature  
  - Label (`0 = Healthy`, `1 = Parkinson`)
- Final dataset:
  - Total samples: **2,559**
  - Parkinson: **1,293 samples**
  - Non-Parkinson: **1,266 samples**
- Saved as a `.pkl` (Pickle) file for future loading.

---

## 🏗️ Model Architectures

### A. Early Fusion

- CNN for image features (4 Conv layers + BatchNorm + MaxPooling)
- MFCC features concatenated with flattened CNN output
- Fully connected layers: 512 → 128 → 1 (binary classification)
- Dropout used to prevent overfitting

### B. Late Fusion

- **LSTM** network processes MFCC features
- **CNN** processes image frames
- Features are merged and passed through a dense layer for classification

---

## 🔁 Training & Evaluation

- Dataset split: 80% training, 20% validation
- Training loop includes:
  - Loss calculation
  - Backpropagation
  - AUC-ROC, Accuracy, Precision, Recall, Sensitivity, Specificity
- Evaluation metrics computed per epoch using validation set

---

## 📊 Performance Metrics

Each model's performance was tracked over epochs:

### 📉 Metrics:

#### Early Fusion  
- Accuracy, AUC-ROC, Confusion Matrix  
- ROC, Loss, Accuracy graphs plotted

#### Late Fusion  
- Accuracy, AUC-ROC, Confusion Matrix  
- ROC, Loss, Accuracy graphs plotted

_(See graphs and full report in PDF or notebook.)_


## 🔖 Keywords

`Parkinson`, `Multimodal`, `Deep Learning`, `CNN`, `LSTM`, `MFCC`, `image Classification`, `Audio Processing`

---

## 📜 License

This project is provided for academic and research purposes.

---

## 👩‍💻 Author

**Sana Kabbani**  
Department of Information Systems Engineering  
Kocaeli University


