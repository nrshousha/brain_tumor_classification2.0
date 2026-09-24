# 🧠 NeuroAI — Brain Tumor Classification & Patient Care Companion

<div align="center">

![NeuroAI Banner](https://img.shields.io/badge/NeuroAI-Brain%20Tumor%20Diagnostics-059669?style=for-the-badge&logo=brain&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-10b981?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-EfficientNetB0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)

**An AI-powered medical diagnostic platform that classifies brain tumors from MRI scans using dual deep learning models simultaneously, coupled with the interactive NeuroCare patient care companion.**

[Quick Start](#-quick-start) • [Architecture](#-system-architecture) • [AI Models](#-ai-models) • [API Reference](#-api-reference) • [Features](#-key-features)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [AI Models](#-ai-models)
- [Project Structure](#-project-structure)
- [Quick Start & Installation](#-quick-start)
- [API Reference](#-api-reference)
- [NeuroCare AI Chatbot](#-neurocare-ai-chatbot)
- [Dataset Details](#-dataset-details)

---

## 🎯 Project Overview

**NeuroAI** is an end-to-end medical AI platform designed to empower clinicians and patients:

1. **Dual-Model MRI Analysis** — Analyzes brain MRI scans simultaneously with two independent deep learning architectures (**PyTorch ResNet-50** and **TensorFlow EfficientNet-B0**), displaying synchronized diagnostic predictions side-by-side for high confidence cross-validation.
2. **Consensus Confidence System** — Automatically detects model agreement. High-confidence consensus is highlighted when both models agree, while potential edge cases trigger specialist consultation warnings.
3. **NeuroCare AI Companion** — An integrated clinical companion that answers patient questions regarding symptoms, side effects, postoperative care, recovery steps, and emergency guidance.
4. **Emerald Cyber-Glassmorphism UI** — A modern, high-tech interface featuring dark emerald glow effects, live drag-and-drop preview, and an interactive floating assistant.

---

## ✨ Key Features

- 🖼️ **Drag & Drop MRI Scanner**: Instant live image preview with support for PNG, JPG, and JPEG formats.
- 🔬 **Synchronized Dual Inference**: In-memory simultaneous execution of PyTorch and TensorFlow models.
- 🎯 **Multi-Class Classification**: Identifies **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor (Healthy)** scans.
- 📊 **Consensus Engine**: Visual badge and confidence indicator comparing model predictions.
- 💬 **NeuroCare AI Floating Assistant**: Real-time conversational health support with quick-prompt suggestions.
- ⚡ **Unified Single-Command Server**: Complete application runs on a single high-performance Flask server.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Client Browser / Web Interface                     │
│                        http://127.0.0.1:5000                            │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Upload MRI Image (POST)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    NeuroAI Unified Server (app.py)                      │
│                                                                         │
│   ┌───────────────────────────┐       ┌─────────────────────────────┐   │
│   │    PyTorch ResNet-50      │       │  TensorFlow EfficientNet-B0 │   │
│   │  (512×512 ImageNet Head)  │       │     (150×150 BGR Tensor)    │   │
│   └─────────────┬─────────────┘       └──────────────┬──────────────┘   │
│                 │                                    │                  │
│                 └─────────────────┬──────────────────┘                  │
│                                   ▼                                     │
│                     Consensus & Prediction Engine                       │
│                     { pytorch_pred, tf_pred }                           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Results View (pred.html)                            │
│    • PyTorch Diagnosis   • TensorFlow Diagnosis   • Consensus Badge     │
│    • Direct Link to NeuroCare AI Assistant with Diagnosis Context       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI Models

### 1. PyTorch ResNet-50
- **Base Architecture**: ResNet-50 (Deep Residual Learning)
- **Classification Head**: Custom FC Network (2048 → 2048 → 4 classes) with `SELU` + `Dropout(0.4)` + `LogSigmoid`
- **Input Resolution**: `512 × 512` px (RGB)
- **Classes**: `['None', 'Meningioma', 'Glioma', 'Pituitary']`
- **Checkpoint**: `assets/bt_resnet50_model.pt`

### 2. TensorFlow EfficientNet-B0
- **Base Architecture**: EfficientNet-B0 (Compound Scaling)
- **Classification Head**: GlobalAveragePooling2D → `Dropout(0.5)` → Dense(4, softmax)
- **Input Resolution**: `150 × 150` px (BGR, raw [0-255] scale)
- **Classes**: `['Glioma', 'No Tumor', 'Meningioma', 'Pituitary']`
- **Model File**: `assets/kaggle_model.h5`

---

## 📁 Project Structure

```bash
brain_tumor_classification2.0/
├── Medical Project/
│   ├── app.py                     # Main Unified Flask Application & AI Server
│   ├── deploy.py                  # Standalone Runner Script
│   ├── requirements.txt           # Python Dependencies (Python 3.10 - 3.12)
│   ├── assets/
│   │   ├── kaggle_model.h5        # TensorFlow EfficientNet-B0 Model
│   │   └── bt_resnet50_model.pt   # PyTorch ResNet-50 Model Checkpoint
│   ├── template/
│   │   ├── index.html             # Scanner & Main Landing Page
│   │   └── pred.html              # Diagnostic Results & Consensus View
│   ├── static/
│   │   ├── css/
│   │   │   └── neuroai.css        # Emerald Bio-Medical Glassmorphic Stylesheet
│   │   └── images/                # UI Assets, Logos & Sample MRI Scans
│   ├── KAN.py                     # Kolmogorov-Arnold Network experimental script
│   └── TFModel_Training.py        # Model Training Pipeline script
├── Data set/                      # MRI Training & Testing Dataset
│   ├── Training/
│   └── Testing/
├── .gitignore                     # Git rules (excludes virtualenvs, temp cache)
└── README.md                      # Comprehensive Project Documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/nrshousha/brain_tumor_classification2.0.git
cd "brain_tumor_classification2.0/Medical Project"
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Web Interface
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 🌐 API Reference

### Programmatic MRI Prediction
You can send MRI scans directly to the REST API for programmatic inference:

- **Endpoint**: `POST /predict`
- **Body**: `multipart/form-data` with key `file`

**Example with cURL**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -F "file=@/path/to/mri_scan.jpg"
```

**JSON Response**:
```json
{
  "pytorch_pred": "Glioma",
  "tf_pred": "Glioma"
}
```

---

### NeuroCare AI Chat Endpoint
- **Endpoint**: `POST /chat`
- **Body**: `multipart/form-data` with key `user_input`

**Example**:
```bash
curl -X POST http://127.0.0.1:5000/chat \
     -F "user_input=What are common recovery steps after treatment?"
```

---

## 💬 NeuroCare AI Chatbot

The **NeuroCare AI Companion** provides patients with instant answers to vital medical questions:
- **Symptoms & Side Effects**: Explains common signs such as chronic headaches, nausea, or visual disturbances.
- **Recovery & Rehabilitation**: Details physical therapy, medication management, and routine follow-up protocols.
- **Emergency Guidance**: Immediate safety instructions for epileptic seizures and acute episodes.
- **Tumor Types**: Comprehensive breakdowns of Gliomas, Meningiomas, and Pituitary tumors.

---

## 📊 Dataset Details

The models were trained on the **Brain Tumor MRI Dataset**, comprising thousands of axial, coronal, and sagittal T1/T2-weighted MRI scans:
- **Glioma**: Infiltrative tumors originating in glial cells.
- **Meningioma**: Typically benign tumors arising from the protective brain meninges.
- **Pituitary**: Glandular neoplasms affecting hormonal homeostasis.
- **No Tumor**: Healthy brain MRI scans.

---

## 📄 License & Disclaimer

This software is developed for research, education, and decision-support purposes. It is not intended as a substitute for professional medical advice, clinical diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for medical conditions.
