# 🧠 NeuroAI — Brain Tumor Classification & Care Companion

<div align="center">

![NeuroAI Banner](https://img.shields.io/badge/NeuroAI-Brain%20Tumor%20Classification-blueviolet?style=for-the-badge&logo=brain)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-EE4C2C?style=for-the-badge&logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-EfficientNetB0-FF6F00?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask)

**An AI-powered web application that classifies brain tumors from MRI scans using two deep learning models simultaneously, paired with an intelligent care chatbot for patient support.**

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [System Architecture](#-system-architecture)
- [AI Models](#-ai-models)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [How to Run](#-how-to-run)
- [NeuroCare Chatbot](#-neurocare-chatbot)
- [Team](#-team)

---

## 🎯 Project Overview

**NeuroAI** is an end-to-end AI medical assistant designed to:

1. **Classify Brain Tumors** — Analyze MRI scans using two independent deep learning models (PyTorch & TensorFlow) and display both results side-by-side for higher diagnostic confidence.
2. **Support Patients** — Provide patients and their families with an AI-powered care chatbot that offers personalized medical guidance, emotional support, and reliable health information.

> This project bridges the gap between cutting-edge AI research and real-world patient care.

---

## ❗ The Problem

Brain tumors affect millions worldwide. Early and accurate detection is critical for patient survival, yet:

- **Radiologists are in short supply** in many hospitals and regions.
- **Misdiagnosis rates** for brain tumors remain concerningly high.
- **Patients feel lost** after diagnosis — unsure about their treatment, what to expect, and where to turn.

---

## ✅ Our Solution

| Challenge | Our Approach |
|-----------|-------------|
| Slow / inaccurate diagnosis | Two AI models analyze each scan independently for **cross-validation** |
| Single-model bias | If both models agree → **high confidence**; if they differ → patient is warned to consult a specialist |
| Patient anxiety | **NeuroCare AI Chatbot** provides 24/7 compassionate, medically-scoped guidance |
| Accessibility | Simple web interface — just upload an MRI image and get results in seconds |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      User (Browser)                          │
│                   http://localhost:3000                       │
└────────────────────────┬─────────────────────────────────────┘
                         │ Upload MRI Image (POST)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Frontend Server — deploy.py (Port 3000)         │
│              Flask Web App │ HTML/CSS/JS UI                  │
└────────────────────────┬─────────────────────────────────────┘
                         │ Forward image to Prediction API
                         ▼
┌──────────────────────────────────────────────────────────────┐
│            Prediction Server — app.py (Port 5000)            │
│                                                              │
│   ┌─────────────────────┐   ┌──────────────────────────┐    │
│   │  PyTorch ResNet-50  │   │  TensorFlow EfficientB0  │    │
│   │  bt_resnet50_model  │   │    kaggle_model.h5        │    │
│   └──────────┬──────────┘   └────────────┬─────────────┘    │
│              └──────────┬────────────────┘                   │
│                         ▼                                    │
│          { pytorch_pred, tf_pred }  (JSON)                   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   Results Page (pred.html)                   │
│       PyTorch Result │ TensorFlow Result │ Consensus Badge   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI Models

### Model 1 — PyTorch ResNet-50

| Property | Details |
|----------|---------|
| **Framework** | PyTorch |
| **Base Architecture** | ResNet-50 (pretrained on ImageNet) |
| **Fine-tuning** | Custom fully-connected head (2048 → 2048 → 4 classes) |
| **Activation** | SELU + LogSigmoid |
| **Input Size** | 512 × 512 px |
| **Output Classes** | None, Meningioma, Glioma, Pituitary |
| **Model File** | `assets/bt_resnet50_model.pt` (~122 MB) |

### Model 2 — TensorFlow EfficientNet-B0

| Property | Details |
|----------|---------|
| **Framework** | TensorFlow / Keras |
| **Base Architecture** | EfficientNet-B0 (pretrained on ImageNet) |
| **Fine-tuning** | GlobalAveragePooling → Dropout(0.5) → Dense(4, softmax) |
| **Input Size** | 150 × 150 px |
| **Output Classes** | Glioma, No Tumor, Meningioma, Pituitary |
| **Training** | 12 epochs, Adam optimizer, Early Stopping + ReduceLR |
| **Model File** | `assets/kaggle_model.h5` (~47 MB) |

### 🔍 Dual-Model Consensus System

The results page intelligently compares both predictions:

- ✅ **Both agree** → High confidence banner displayed
- ⚠️ **Models differ** → Warning banner urges specialist consultation

---

## ✨ Key Features

- 🖼️ **Drag & Drop MRI Upload** with live image preview
- 🔬 **Dual-Model Analysis** — PyTorch + TensorFlow simultaneously
- 🎨 **Color-coded Diagnoses** — each tumor type has a unique color
- 🤝 **Consensus Banner** — smart agreement/disagreement detection
- 🧠 **NeuroCare Chatbot** — AI companion for patients (powered by LLaMA 3)
- 🌐 **Web Search Integration** — chatbot fetches real-time medical info when needed
- 📱 **Fully Responsive** — works on desktop and mobile

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JavaScript |
| **Backend** | Python, Flask |
| **ML Framework 1** | PyTorch, TorchVision |
| **ML Framework 2** | TensorFlow, Keras |
| **Image Processing** | OpenCV, Pillow (PIL) |
| **Chatbot** | LLaMA 3 (8B) via Open-WebUI / Groq API |
| **Fonts** | Google Fonts (Inter, Space Grotesk) |
| **Server** | Gunicorn (production) |

---

## 📁 Project Structure

```
Medical Project/
│
├── 📄 app.py                    # Prediction API server (Port 5000)
│                                # Loads & runs both AI models
│
├── 📄 deploy.py                 # Frontend Flask server (Port 3000)
│                                # Handles uploads & session management
│
├── 📄 TFModel_Training.py       # TensorFlow EfficientNetB0 training script
│
├── 📄 KAN.py                    # Kolmogorov-Arnold Network experiment
│
├── 📁 template/
│   ├── 🌐 index.html            # Upload page (drag & drop MRI)
│   └── 🌐 pred.html             # Results page (dual model output)
│
├── 📁 static/
│   ├── 📁 css/
│   │   └── neuroai.css          # Shared stylesheet for all pages
│   └── 📁 images/               # UI assets (logos, brain icon)
│
├── 📁 assets/
│   ├── 🧠 bt_resnet50_model.pt  # PyTorch trained model (~122 MB)
│   └── 🧠 kaggle_model.h5       # TensorFlow trained model (~47 MB)
│
├── 📁 Home Page/
│   └── 🌐 index.html            # Project landing page
│
├── 📁 Data set/                 # MRI training & testing images
│
├── 📄 requirements_deploy.txt   # Minimal dependencies for deployment
├── 📄 requirements.txt          # Full development dependencies
└── 📄 README.md                 # This file
```

---

## 📊 Dataset

**Brain Tumor Classification (MRI)** — Kaggle

| Property | Details |
|----------|---------|
| **Source** | [Kaggle Dataset by Sartaj Bhuvaji](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) |
| **Classes** | Glioma Tumor, Meningioma Tumor, Pituitary Tumor, No Tumor |
| **Total Images** | ~3,264 MRI scans |
| **Split** | Training + Testing folders |
| **Format** | JPG images |

---

## 🚀 How to Run

### Prerequisites
- Python 3.10
- Conda environment named `classification`
- GPU (optional — CPU is supported)

### 1. Clone & Install

```bash
git clone <repository_url>
cd "Medical Project"

# Install dependencies
pip install -r requirements_deploy.txt

# Fix protobuf version conflict
pip install protobuf==3.20.3
```

### 2. Download Models

Place the models in the `assets/` folder:

| Model | Download Link |
|-------|--------------|
| PyTorch ResNet-50 | [Download](https://drive.google.com/file/d/1LbFYQWl-gsi9tKo6SDNIwRTd0yi88GAJ/view?usp=sharing) |
| TensorFlow EfficientNetB0 | [Download](https://drive.google.com/file/d/1nCbXHx2LMmgRByJ8mG7OtPP0ZsSJUje-/view?usp=sharing) |

### 3. Run the Servers

Open **two terminals** and run:

```bash
# Terminal 1 — AI Prediction Server
conda activate classification
python app.py
# → Running on http://localhost:5000
```

```bash
# Terminal 2 — Web Frontend
conda activate classification
python deploy.py
# → Running on http://localhost:3000
```

### 4. Open the App

Navigate to **http://localhost:3000** and upload any brain MRI image!

---

## 🤖 NeuroCare Chatbot

The integrated chatbot is powered by **LLaMA 3 (8B)** and is specifically designed for brain tumor patients.

### What it does:
- 💬 Explains tumor types, treatments, and medical terms in simple language
- 🧘 Provides mental health support and coping strategies
- 🔍 Searches the web for the latest medical research when needed
- 🚨 Alerts users to seek emergency care for severe symptoms
- 🚫 Strictly stays within the medical domain (no off-topic questions)

### Setup Options:

**Option A — Open-WebUI (Local LLaMA 3)**
```bash
pip install open-webui
open-webui serve
# → Running on http://localhost:8080
```

**Option B — Groq API (Faster, Cloud)**
1. Get a free API key from [Groq Console](https://console.groq.com/keys)
2. In Open-WebUI → Settings → Admin Settings → Connections
3. Add: `https://api.groq.com/openai/v1` with your key

---

## 👥 Team

**NeuroAI Detectives** — Built with ❤️ for patients everywhere.

> *"Using technology to bring hope and clarity to one of medicine's most challenging diagnoses."*

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## 🙏 Acknowledgments

- PyTorch & TensorFlow communities
- Open-WebUI & Groq teams
- Kaggle dataset contributors
- Medical professionals who inspired the care chatbot vision
