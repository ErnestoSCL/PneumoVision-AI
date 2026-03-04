# 🫁 PneumoVision AI: Pneumonia Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Gradio-UI-FF4B4B?logo=gradio" alt="Gradio">
  <img src="https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface" alt="HF Spaces">
</p>

> **PneumoVision AI** is an intuitive, AI-powered clinical dashboard designed to assist medical professionals in the rapid and preliminary detection of pneumonia from chest X-ray images.

### 🌐 [Live Demo en Hugging Face](https://huggingface.co/spaces/ErnestoRepos/neumonia_prediction)

---

## ✨ Features

- **Deep Learning Model:** Utilizes a Convolutional Neural Network (CNN) trained on chest X-rays to distinguish between normal lungs and pneumonia-affected lungs.
- **Clinical Dashboard:** Features a clean, medical-grade web interface powered by Gradio Blocks.
- **Adjustable Sensitivity Threshold:** Allows medical personnel to fine-tune the detection alarm threshold to balance sensitivity and specificity according to clinical needs.
- **Patient Context Integration:** Optionally accepts patient data (name, age, symptoms) to generate a consolidated preliminary report.
- **Real-Time Inference:** Instant visual feedback and confidence scores.

---

## 🛠️ Stack Tecnológico

| Herramienta | Uso |
|-------------|-----|
| **TensorFlow & Keras** | Inferencia del modelo CNN y procesamiento |
| **Gradio** | Construcción de la interfaz y Dashboard Web |
| **Python** | Lenguaje Principal |
| **Pillow & NumPy** | Carga, normalización y preprocesamiento de imágenes |

---

## 🚀 Cómo correr localmente

### Prerrequisitos
Deberás tener **Python 3.8+** instalado.

### Instalación

```bash
# 1. Clonar este repositorio:
git clone https://github.com/ErnestoSCL/PneumoVision-AI.git
cd PneumoVision-AI

# 2. Instalar las dependencias requeridas (Keras, TensorFlow, Gradio):
pip install -r requirements.txt

# 3. Lanzar la aplicación
python app.py
```

> **Abre tu navegador:** Ve a la URL local generada (usualmente `http://127.0.0.1:7860`).

---

## ⚠️ Disclaimer Médico
*Reconocimiento médico automatizado: Esta herramienta es un prototipo y un asistente preliminar de IA. No está concebida para reemplazar el criterio, diagnóstico, o tratamiento de un profesional de la salud. Acuda siempre a un radiólogo o médico calificado.*
