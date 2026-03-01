# 🫁 PneumoVision AI: Pediatric Pneumonia Detector

PneumoVision AI is an intuitive, AI-powered clinical dashboard designed to assist medical professionals in the rapid and preliminary detection of pneumonia from chest X-ray images.

## ✨ Features
- **Deep Learning Model:** Utilizes a Convolutional Neural Network (CNN) trained on chest X-rays to distinguish between normal lungs and pneumonia-affected lungs.
- **Clinical Dashboard:** Features a clean, medical-grade web interface powered by Gradio Blocks.
- **Adjustable Sensitivity Threshold:** Allows medical personnel to fine-tune the detection alarm threshold to balance sensitivity and specificity according to clinical needs.
- **Patient Context Integration:** Optionally accepts patient data (name, age, symptoms) to generate a consolidated preliminary report.
- **Real-Time Inference:** Instant visual feedback and confidence scores.

## 🛠️ Technologies Used
- **TensorFlow & Keras:** For CNN model inference and image array processing.
- **Python 3:** Core programming language.
- **Gradio:** For building the interactive web-based clinical dashboard.
- **Pillow (PIL) & NumPy:** For image loading, normalization, and pre-processing.

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python 3.8+ installed. You will also need `pip` to install dependencies.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ErnestoSCL/PneumoVision-AI.git
   cd PneumoVision-AI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `tensorflow` and `gradio` installed)*

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

## ⚠️ Disclaimer
*This tool is a prototype and a preliminary AI-assisted analysis tool. It is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified radiologist or physician.*
