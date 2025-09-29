import os
import numpy as np
import cv2
import base64
import requests
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CONFIGURATION ---
MODEL_PATH = 'cardiomyopathy_classifier_v1.keras' 
IMG_SIZE = 128

# The class names must be in the same order as your training data's labels
CLASS_NAMES = ['Normal', 'Hypertrophic (HCM)', 'Dilated (DCM)', 'Infarction (MINF)']

# --- INITIALIZE THE FLASK APP ---
app = Flask(__name__)
CORS(app) 

# --- LOAD THE TRAINED MODEL ---
print(f"[*] Loading classifier model from: {MODEL_PATH}")
try:
    classifier_model = load_model(MODEL_PATH)
except Exception as e:
    print(f"[!] Error loading classifier model: {e}")
    classifier_model = None

# --- GEMINI API FUNCTION ---
def get_gemini_description(image_bytes, classification_result):
    """
    Sends the image and classification to Gemini for a detailed description.
    """
    print("[*] Getting description from Gemini...")
    
    # In a real app, use a secure way to handle API keys.
    # For Colab/Canvas, this key is often handled by the environment.
    api_key = "AIzaSyCcyX9oPt9g3b8WFB1reRxvS1shF6dKVkE" # This will be provided by the execution environment.
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    # Encode the image bytes to base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # --- THIS IS THE UPDATED PROMPT ---
    # It now specifically asks Gemini to describe visible features.
    prompt = (
        "You are a helpful medical imaging assistant. The provided cardiac MRI scan has been "
        f"classified by another AI model as '{classification_result}'. "
        "Analyze the image and provide a brief, one-paragraph educational description of the visible condition of the heart. "
        "Specifically mention any observable features like a dilated ventricle, thickened muscle, or potential areas of damage that are characteristic of this condition. "
        "Do not offer a diagnosis or medical advice. Frame your description as an observation of the image itself. "
        "Explain it in simple terms for a non-expert."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status() # Raise an exception for bad status codes
        
        result = response.json()
        
        if 'candidates' in result and result['candidates']:
            description = result['candidates'][0]['content']['parts'][0]['text']
            return description.strip()
        else:
            # Handle cases where the response structure is unexpected
            print(f"[!] Gemini response format error: {result}")
            return "Could not generate a detailed description at this time."

    except requests.exceptions.RequestException as e:
        print(f"[!] Error calling Gemini API: {e}")
        return "Error connecting to the analysis service."
    except Exception as e:
        print(f"[!] An unexpected error occurred in get_gemini_description: {e}")
        return "An unexpected error occurred during analysis."


# --- PREPROCESSING FUNCTION ---
def preprocess_image(image_bytes):
    try:
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if np.max(img_resized) > 0:
            img_normalized = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized))
        else:
            img_normalized = img_resized
        img_3_channel = np.stack((img_normalized,)*3, axis=-1)
        img_final = np.expand_dims(img_3_channel, axis=0)
        
        return img_final
    except Exception as e:
        print(f"[!] Error preprocessing image: {e}")
        return None

# --- PREDICTION API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    if classifier_model is None:
        return jsonify({'error': 'Classifier model not loaded. Check server logs.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        image_bytes = file.read()
        
        # Preprocess for our classifier
        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            return jsonify({'error': 'Failed to process the image'}), 500

        # Step 1: Get classification from our trained model
        prediction_probs = classifier_model.predict(processed_image)[0]
        predicted_class_index = np.argmax(prediction_probs)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(prediction_probs[predicted_class_index])

        # Step 2: Get detailed description from Gemini
        # We need to rewind the file stream to pass the original bytes to Gemini
        description = get_gemini_description(image_bytes, predicted_class_name)
        
        # Step 3: Return both results
        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f"{confidence*100:.2f}%",
            'description': description 
        })

# --- MAIN ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

