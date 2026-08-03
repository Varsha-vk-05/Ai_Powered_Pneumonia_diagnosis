
**Ai_Powered_Pneumonia_diagnosis**.
Just copy-paste into your `README.md` (or tell me to push directly).

---

# 🫁 AI-Powered Pneumonia Diagnosis System

### A Multi-Test Medical Diagnostic Web + Android Application

**Built using Deep Learning (PyTorch), Flask, Vite Frontend, Real-Time Voice Interaction, and Region-Wise Infection Analysis**

---

## ⭐ Overview

The **AI-Powered Pneumonia Diagnosis System** is an end-to-end medical diagnostic platform capable of identifying pneumonia (Normal / Bacterial / Viral) using:

* **Chest X-ray Images**
* **CT Scan Images**
* **Blood Test Data**
* **Pulse Oximetry**
* **Sputum Tests**
* **Bronchoscopy Reports**

The system supports:


* 🖥️ **Web Application (Vite + Tailwind + JS)**
* 🤖 **Flask AI Backend**
* 🔥 **Region-Wise Grad-CAM Infection Highlighting**
* 📊 **Infection Percentage Calculation**
* 🎙️ **Voice Input + Voice Output (Multi-Language)**
* 🌙 **Dark/Light Mode**
* 🌐 **Multi-Language Support (English, Hindi, Tamil & Kannada)**

This project brings AI-driven pneumonia diagnosis directly to patients, clinics, and rural healthcare centers.

---

# 🚀 Features

### 🩺 **1. AI Model (PyTorch – ResNet-18)**

* Classifies **Normal, Bacterial, Viral Pneumonia**
* Supports **X-Ray + CT Scan**
* Integrated **Grad-CAM** for explainable predictions
* Infection localization with **infected region percentage**

---

### 🌐 **2. Web Frontend (Vite + Tailwind)**

* Clean, fast UI
* Image upload preview
* Multi-test support
* Voice input
* Result cards with animations
* Dark/Light theme switcher
* Multi-language UI
* Mobile-responsive layout

---


### 🔗 **4. Flask Backend**

* `/predict` endpoint accepts multiple test types
* Parses medical reports (text/JSON)
* Runs model inference
* Generates Grad-CAM heatmaps
* Sends JSON response + infection percentage
* CORS-enabled for Android + Web

---

### 🧠 **5. Explainability**

* Grad-CAM Heatmap
* Region-wise mask
* Infection percentage calculation
* Localized region (Left lung / Right lung / Both)

---

## 📂 Project Structure

```
Ai_Powered_Pneumonia_diagnosis/
│
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── utils/
│   │   ├── gradcam.py
│   │   ├── preprocessing.py
│   │   └── infection_percentage.py
│   ├── static/ (generated Grad-CAM images)
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── styles/
│   ├── public/
│   └── vite.config.js
│
├── android/
│   ├── app/
│   │   ├── src/main/java/com/pneumoniaai/
│   │   ├── res/
│   │   └── AndroidManifest.xml
│
├── models/
│   ├── model.pth
│   └── label_map.json
│
└── README.md
```

---

# ⚙️ Installation & Setup

## 📌 1. Clone the repository

```bash
git clone https://github.com/Varsha-vk-05/Ai_Powered_Pneumonia_diagnosis.git
cd Ai_Powered_Pneumonia_diagnosis
```

---

# 🧠 Backend Setup (Flask + PyTorch)

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # Windows
```

## 2️⃣ Install Dependencies

```bash
pip install -r backend/requirements.txt
```

## 3️⃣ Run the Server

```bash
cd backend
python app.py
```

The backend will start on:

```
http://127.0.0.1:5000
```

---

# 🖥️ Frontend Setup (Vite + Tailwind + JS)

## 1️⃣ Install Node Modules

```bash
cd frontend
npm install
```

## 2️⃣ Start Frontend

```bash
npm run dev
```

---

```
file: image or report
type: "xray" | "ct" | "blood" | "sputum" | "bronchoscopy" | "pulse"
```

#### Response:

```json
{
  "class": "Bacterial Pneumonia",
  "confidence": 94.87,
  "infection_percentage": 62.1,
  "infected_region": "Right Lung",
  "gradcam_url": "/static/result_12345.png"
}
```

---

# 📊 Infection Percentage Calculation

The AI automatically:

* Segments lung region
* Compares infected pixels vs healthy pixels
* Calculates:

```
infection% = (infected pixels / total lung pixels) × 100
``


# 🏁 Conclusion

This project provides a **complete, production-ready medical AI system**, combining deep learning, Web,and explainable AI. It significantly improves pneumonia diagnosis through automation, accuracy, and accessibility.

---

# 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

# 📜 License

MIT License © 2025 Varsha S



Just tell me **"add badges"**, **"add diagrams"**, or **"generate screenshots"**.
