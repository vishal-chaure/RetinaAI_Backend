import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import logging
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
import base64
import io
from PIL import Image
import datetime
import requests
from pathlib import Path

# Configure TensorFlow to use CPU and suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

# Configure TensorFlow for CPU optimization
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Disable GPU
tf.config.set_visible_devices([], 'GPU')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Hugging Face model URL (update this with your actual link)
MODEL_URL = "https://huggingface.co/your-username/your-model-repo/resolve/main/fine_tuned_resnet50_model.h5"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fine_tuned_resnet50_model.h5")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if not present
try:
    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found locally. Downloading from Hugging Face...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Model downloaded successfully!")

    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully from downloaded file!")
except Exception as e:
    logger.error(f"Error downloading or loading model: {str(e)}")
    raise

# Get last conv layer name
last_conv_layer = [layer.name for layer in model.layers if 'conv' in layer.name][-1]

# Grad-CAM setup
def get_score_function(pred_idx):
    return CategoricalScore([pred_idx])
gradcam = Gradcam(model, clone=True)

def generate_rule_based_explanation(pred_idx, confidence, focus_area="highlighted region"):
    class_explanations = {
        0: f"The model did not detect any abnormal signs such as microaneurysms, exudates, or hemorrhages. "
           f"It focused mainly on the {focus_area}, suggesting a healthy retina with **No Diabetic Retinopathy** (Class 0).",
        1: f"The model detected early signs such as a few microaneurysms in the {focus_area}. "
           f"This suggests **Mild DR** (Class 1).",
        2: f"The model found several lesions (microaneurysms, hemorrhages) in the {focus_area}, "
           f"indicating **Moderate DR** (Class 2).",
        3: f"The model focused on widespread hemorrhages and abnormalities in the {focus_area}, "
           f"suggesting **Severe DR** (Class 3).",
        4: f"The model detected abnormal blood vessel growth (neovascularization) or large lesions in the {focus_area}, "
           f"consistent with **Proliferative DR** (Class 4)."
    }

    base = class_explanations.get(pred_idx, "Unknown class.")
    return f"📊 Prediction: Class {pred_idx}\n\n" \
           f"✅ Confidence: {confidence * 100:.2f}%\n\n\n" \
           f"💬 Explanation: {base}"

def process_image(image_data):
    try:
        # Convert base64 to image
        image_data = image_data.split(',')[1]  # Remove the data URL prefix
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Resize and preprocess
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        
        # Get prediction
        preds = model.predict(x)
        pred_idx = np.argmax(preds)
        confidence = preds[0][pred_idx]

        # Generate Grad-CAM
        cam = gradcam(score=get_score_function(pred_idx),
                      seed_input=x,
                      penultimate_layer=last_conv_layer)
        heatmap = np.uint8(255 * cam[0])

        # Convert heatmap to base64
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return {
            'prediction_class': int(pred_idx),
            'confidence': float(confidence),
            'explanation': generate_rule_based_explanation(pred_idx, confidence),
            'gradcam_image': f'data:image/png;base64,{heatmap_base64}',
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

@app.route('/')
def home():
    return jsonify({
        'status': 'API is running',
        'endpoints': {
            'predict': '/predict (POST) - Upload retinal image for analysis',
            'health': '/health (GET) - Check API health status'
        },
        'usage': 'Send a POST request to /predict with a JSON body containing base64 encoded image in the "image" field'
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        image_data = data['image']
        result = process_image(image_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 