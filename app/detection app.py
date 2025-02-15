from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import json
import time

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = r'...../app/static/uploads'
MODEL_PATH = r'..../app/trained_model/prediction.h5'
CLASS_INDICES_PATH = r'...../app/class_indices.json'

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)
    class_names = {int(k): v for k, v in class_indices.items()}  # Ensure keys are integers

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Adjust according to your model's input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    return render_template('hero1.html')

@app.route('/submit')
def submit():
    return render_template('submit.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Rate limiting logic (this is basic and should be improved for production use)
    current_time = int(time.time())
    rate_limit_reset = int(request.headers.get('X-RateLimit-Reset', 0))
    if current_time - rate_limit_reset < 60:
        return jsonify({'error': 'Rate limit exceeded'}), 429

    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if image:
        filename = secure_filename(image.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)

        try:
            # Preprocess the image
            image_array = preprocess_image(file_path)

            # Predict the image
            prediction = model.predict(image_array)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]

            # Pass the result and image file path to the template
            return render_template('final.html', result=predicted_class, image_file=filename)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(debug=True)
