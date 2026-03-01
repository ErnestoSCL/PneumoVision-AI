import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import gradio as gr
import os

# --- Configuración de Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'cnn_neumonia.keras')
IMG_HEIGHT = 64
IMG_WIDTH = 64

# --- 1. Cargar el modelo con validación ---
try:
    model = keras.models.load_model(model_path)
    print(f"✅ Modelo cargado exitosamente desde: {model_path}")
except Exception as e:
    print(f"❌ Error crítico al cargar el modelo: {e}")
    print("Asegúrate de que el archivo 'cnn_neumonia.keras' esté en la misma carpeta que este script.")
    model = None # Prevención de cuelgues si no encuentra el modelo

# --- 2. Funciones de Procesamiento ---
def preprocess_image_for_prediction(image: Image.Image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0                
    return img_array

def analyze_xray(image, patient_name, patient_age, symptoms, threshold):
    if image is None:
        return None, "⚠️ Por favor, carga una imagen de radiografía primero."
    if model is None:
        return None, "⚠️ El modelo no está cargado correctamente."

    processed_image = preprocess_image_for_prediction(image)
    prediction = model.predict(processed_image, verbose=0)
    
    # Asumiendo validación binaria (salida sigmoide)
    probability_pneumonia = float(prediction[0][0])
    probability_normal = 1.0 - probability_pneumonia
    
    # Diagnóstico basado en el umbral personalizado
    if probability_pneumonia >= threshold:
        diagnosis = "NEUMONÍA DETECTADA"
        color = "#ef4444" # Rojo
        # Forzar que visualmente la barra de Neumonía gane en el panel top_classes de Gradio
        display_pneumonia = max(probability_pneumonia, 0.51)
        display_normal = 1.0 - display_pneumonia
    else:
        diagnosis = "NORMAL (Sin evidencia clara de neumonía)"
        color = "#22c55e" # Verde
        # Forzar que visualmente la barra de Normal gane en el panel top_classes de Gradio
        display_normal = max(probability_normal, 0.51)
        display_pneumonia = 1.0 - display_normal
        
    # Formateo de las salidas visuales usando los valores recalibrados para sincronía visual
    confidences = {
        "Neumonía": display_pneumonia,
        "Normal": display_normal
    }
        
    # Generación de informe
    report = f"### Informe Radiológico Preliminar\n"
    report += f"**Paciente:** {patient_name if patient_name else 'No especificado'} | **Edad:** {patient_age if patient_age else 'N/A'}\n"
    report += f"**Síntomas:** {symptoms if symptoms else 'Ninguno indicado'}\n\n"
    report += f"---\n\n"
    report += f"**Diagnóstico Asistido:** <span style='color:{color}; font-weight:bold; font-size:1.1em;'>{diagnosis}</span>\n\n"
    report += f"**Confianza del modelo (Neumonía):** {probability_pneumonia:.2%}\n"
    report += f"**Umbral utilizado:** {threshold:.2f}\n\n"
    report += f"> *Aviso: Este es un análisis preliminar generado por IA y no sustituye el criterio de un médico radiólogo profesional.*"

    return confidences, report

# --- 3. Interfaz de Gradio (Usando Blocks para un Dashboard Médico) ---

# Preparamos variables de ejemplo
example_list = [
    [os.path.join(BASE_DIR, "pneumonia1.jpeg"), "Juan Pérez", 45, "Tos con flema, fiebre alta", 0.5],
    [os.path.join(BASE_DIR, "normal1.jpeg"), "María García", 30, "Chequeo de rutina", 0.5],
    [os.path.join(BASE_DIR, "pneumonia2.jpeg"), "Carlos López", 60, "Dificultad para respirar", 0.5],
]
existing_examples = [ex for ex in example_list if os.path.exists(ex[0])]

# Tema visual suave y clínico
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

# CSS personalizado para centrar y darle una apariencia más compacta profesional
custom_css = """
/* Limitar el ancho máximo de la aplicación y centrarla */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding-top: 2rem !important;
}

/* Forzar que la imagen no se vea aplastada y mantenga su proporción natural */
#custom-image img {
    object-fit: contain !important;
    max-height: 450px !important;
}

/* Separación del contenedor del informe con padding y fondo translúcido */
#report-box {
    padding: 1.5rem !important;
    background-color: rgba(30, 41, 59, 0.3) !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    margin-top: 1rem !important;
}

/* Ocultar el texto duro en ingles "Examples" generado por Gradio */
#casos-prueba-box > div > span, #casos-prueba-box .label-text {
    display: none !important;
}

/* Estilizar el pie de página profesional */
.footer-text {
    text-align: center;
    color: #64748b;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e2e8f0;
    font-size: 0.95em;
    line-height: 1.6;
}
.footer-text a {
    color: #3b82f6;
    text-decoration: none;
    font-weight: 500;
}
.footer-text a:hover {
    text-decoration: underline;
}
"""

with gr.Blocks(theme=theme, css=custom_css, title="AI Neumonía Detector") as iface:
    gr.Markdown(
        """
        # Asistente de Diagnóstico de Neumonía por Radiografía
        Esta herramienta de apoyo clínico utiliza Deep Learning para analizar radiografías de tórax (CXR) estructurales y estimar la probabilidad de neumonía. 
        """
    )
    
    with gr.Row(equal_height=False):
        # Columna Izquierda: Imagen y Controles
        with gr.Column(scale=4):
            gr.Markdown("### 1. Imagen Radiológica")
            image_input = gr.Image(type="pil", label="Cargar Radiografía de Tórax", elem_id="custom-image")
            
            with gr.Accordion("Configuración Avanzada", open=False):
                gr.Markdown("Un umbral más bajo detectará más neumonías (alta sensibilidad), pero puede dar falsos positivos.")
                threshold_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, 
                                             label="Umbral de Alarma de Neumonía")
            
            analyze_btn = gr.Button("Analizar Radiografía", variant="primary", size="lg")
            
        # Columna Derecha: Paciente y Resultados
        with gr.Column(scale=5):
            gr.Markdown("### 2. Datos Clínicos (Opcional)")
            with gr.Row():
                patient_name = gr.Textbox(label="Nombre del Paciente", placeholder="Ej. Juan Pérez")
                patient_age = gr.Number(label="Edad", precision=0)
            symptoms = gr.Textbox(label="Síntomas principales", lines=2)
            
            gr.Markdown("### 3. Resultados del Análisis por IA")
            with gr.Group():
                label_output = gr.Label(label="Diagnóstico Visual (Ajustado por Umbral)", num_top_classes=2)
                report_output = gr.Markdown(label="Informe Generado", elem_id="report-box")

    # Acciones de la interfaz
    analyze_btn.click(
        fn=analyze_xray,
        inputs=[image_input, patient_name, patient_age, symptoms, threshold_slider],
        outputs=[label_output, report_output]
    )
    
    if existing_examples:
        gr.Markdown("### Casos de Prueba")
        gr.Examples(
            examples=existing_examples,
            inputs=[image_input, patient_name, patient_age, symptoms, threshold_slider],
            outputs=[label_output, report_output],
            fn=analyze_xray,
            cache_examples=False,
            elem_id="casos-prueba-box",
            label=""
        )

    # --- Footer Profesional ---
    gr.HTML(
        """
        <div class="footer-text">
            <strong>Desarrollado por:</strong> Ernesto Castro Lozano<br>
            Contacto: <a href="mailto:ernestosaniel123@gmail.com">ernestosaniel123@gmail.com</a> | 
            LinkedIn: <a href="https://www.linkedin.com/in/ernesto-saniel-castro-lozano-96ba2b268" target="_blank">Ernesto Saniel Castro Lozano</a>
        </div>
        """
    )
    
# --- 4. Lanzamiento ---
if __name__ == "__main__":
    iface.launch(debug=True, share=True) # Mantengo share=True como pedías antes para links públicos