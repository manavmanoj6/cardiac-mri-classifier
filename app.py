import os
import io
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import google.generativeai as genai

# --- Configuration ---
# IMPORTANT: You must set your GOOGLE_API_KEY as an environment variable in Render.
# In your Render dashboard -> Environment -> Add Environment Variable
# Key: GOOGLE_API_KEY
# Value: your_actual_google_api_key
# For local testing, you can temporarily set it here, but DO NOT commit it to GitHub.
# GOOGLE_API_KEY = "YOUR_API_KEY_HERE" 
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("[Warning] GOOGLE_API_KEY environment variable not set. Gemini features will be disabled.")


# --- Initialize Flask App ---
# The `static_folder` and `template_folder` are set to handle the Dimension template structure
app = Flask(__name__, static_folder='assets', template_folder='.')
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Load the Trained Model ---
try:
    # Ensure the model file is in the same directory as app.py
    model = tf.keras.models.load_model('cardiomyopathy_classifier_v1.keras')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Gemini Model Initialization ---
gemini_model = None
if GOOGLE_API_KEY:
    try:
        generation_config = {
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 1024,
        }
        gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        print("Gemini model initialized successfully!")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")


# --- Class Names and Fallback Descriptions ---
CLASS_NAMES = ['Normal', 'Hypertrophic (HCM)', 'Dilated (DCM)', 'Infarction (MINF)']
DESCRIPTIONS = {
    'Normal': "The heart structure appears to be within normal limits.",
    'Hypertrophic (HCM)': "Hypertrophic Cardiomyopathy is characterized by a thickening of the heart muscle.",
    'Dilated (DCM)': "Dilated Cardiomyopathy is characterized by an enlarged and weakened left ventricle.",
    'Infarction (MINF)': "Myocardial Infarction (heart attack) can result in scarring and damage to the heart muscle."
}

# --- Helper Functions ---
def preprocess_image(image_file):
    try:
        in_memory_file = io.BytesIO()
        image_file.save(in_memory_file)
        in_memory_file.seek(0)
        image = Image.open(in_memory_file).convert('L') # Convert to grayscale
        image_np = np.array(image)
        
        resized_img = cv2.resize(image_np, (128, 128))
        if np.max(resized_img) > 0:
            normalized_img = resized_img / 255.0
        else:
            normalized_img = resized_img

        img_3_channel = np.stack((normalized_img,) * 3, axis=-1)
        return np.expand_dims(img_3_channel, axis=0)
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# --- Routes ---
@app.route('/')
def index():
    # This route serves your main HTML file.
    # Flask will look for 'index.html' in the 'template_folder' we defined above.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    processed_image = preprocess_image(file)
    if processed_image is None:
        return jsonify({'error': 'Could not process image'}), 500

    # 1. Get prediction from your trained model
    prediction_probs = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction_probs)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = f"{np.max(prediction_probs) * 100:.2f}%"

    # 2. Get description from Gemini (if available)
    description = DESCRIPTIONS.get(predicted_class_name, "No description available.") # Fallback
    
    if gemini_model:
        try:
            file.seek(0) 
            image_for_gemini = Image.open(file)
            
            prompt = (
                "You are a helpful medical imaging assistant. "
                f"A deep learning model has classified this cardiac MRI scan as '{predicted_class_name}'. "
                "Based on the visual evidence in the image, provide a brief, one-paragraph educational description for a non-expert. "
                "Describe the specific visual features of the heart in the image (like ventricle size, wall thickness, or visible damage) that are consistent with this classification."
            )
            
            response = gemini_model.generate_content([prompt, image_for_gemini])
            if response.text:
                description = response.text

        except Exception as e:
            print(f"Gemini API error: {e}")
            # If Gemini fails, we will still return the prediction with the fallback description.

    return jsonify({
        'prediction': predicted_class_name,
        'confidence': confidence,
        'description': description
    })

# --- Main entry point for deployment ---
if __name__ == '__main__':
    # Get port from environment variable for Render, default to 5000 for local dev
    port = int(os.environ.get('PORT', 5000))
    # '0.0.0.0' is required to be accessible externally on Render
    app.run(host='0.0.0.0', port=port)

