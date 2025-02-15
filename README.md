# Plant Disease Detection using Deep Learning

## 🌱 Overview
This project is a **Plant Disease Detection System** that utilizes **Deep Learning (CNN)** to classify plant diseases based on leaf images. The model is trained on a dataset of plant images and deployed for real-time detection.

## 🛠️ Technologies Used
- **Python**
- **TensorFlow/Keras**
- **NumPy & Matplotlib**
- **PIL (Python Imaging Library)**
- **ImageDataGenerator** for data preprocessing

## 📂 Dataset
The dataset is stored in the directory:
```
C:/Users/Ajmal/Documents/programs/plant/plant dataset
```
It is split into **training (80%)** and **validation (20%)** sets using `ImageDataGenerator`.

## 🔧 Model Architecture
The model is a **Convolutional Neural Network (CNN)** consisting of:
- **Conv2D & MaxPooling layers** for feature extraction
- **Flatten & Dense layers** for classification
- **Softmax activation** for multi-class disease prediction

## 🚀 How to Run
### 1️⃣ Install Dependencies
Ensure you have the required Python libraries:
```bash
pip install tensorflow numpy matplotlib pillow
```

### 2️⃣ Train the Model
Run the Python script to train the model:
```bash
python train.py
```
The training process includes:
- **Data Augmentation**
- **5 Epochs** for model training
- **Validation Accuracy Calculation**

### 3️⃣ Evaluate the Model
The model will be evaluated using validation data, and accuracy metrics will be displayed.

### 4️⃣ Make Predictions
You can predict plant disease by using:
```python
predicted_class = predict_image_class(model, "path_to_image.jpg", class_indices)
print("Predicted Disease:", predicted_class)
```

## 📊 Training Results
- The model plots **Accuracy** and **Loss** graphs to visualize performance.
- The trained model is saved as `prediction.h5`.

## 📜 License
This project is **open-source** and available for further development!

---
🔗 **Contributor1:** [Mazin Muneer](https://github.com/Sha-330)
🔗 **Contributor2:** [Ajmal Shan](https://github.com/Sha-330)
🔗 **Contributor2:** [Adam Nahan](https://github.com/Sha-330)

