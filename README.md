# Plant Disease Detection System

## Overview
The **Plant Disease Detection System** is a machine learning-based application designed to detect diseases in plants from images. This system uses a **Convolutional Neural Network (CNN)** model trained on a dataset of plant images to classify diseases and provide accurate predictions.

## Features
- **Image Upload**: Users can upload images of plants to the web application.
- **Disease Detection**: The application uses a pre-trained CNN model to predict the disease in the uploaded image.
- **Result Display**: The predicted disease is displayed along with the uploaded image.

## Installation
### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install tensorflow flask numpy pillow
```
##Model Training
Dataset
The dataset is stored in directories for training and validation.

Each subdirectory represents a different plant disease class.

Model Training
A CNN-based model is trained using tensorflow.keras.

The trained model is saved as prediction.h5.

Class indices are saved in class_indices.json.

##Code Snippet
```bash
model.save("prediction.h5")
print("Model saved successfully!")
```
Usage
Running the Application
Start the Flask server:

```bash
python detection_app.py
Access the application:
```

###Open your web browser and go to http://127.0.0.1:5000/.

You should see the home page of the application.

Uploading an Image
Navigate to the "Submit" page.

Upload an image of a plant using the file upload form.

##Viewing the Result
After uploading the image, the application will process it and display the predicted disease along with the uploaded image.

##Results & Evaluation
Model Accuracy: The model's validation accuracy is displayed during training.

Prediction: The predicted disease class is shown along with the uploaded image.

## Contributors
- **Mazin Muneer (mazi)** - [GitHub](https://github.com/Sha-330) | [LinkedIn](https://www.linkedin.com/in/mazin-muneer?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BdvLryXBiQjypj5RZtQSCow%3D%3D)
- **Ajmal Shan. P (Sha)** - [GitHub](https://github.com/Sha-330) | [LinkedIn](https://www.linkedin.com/in/ajmal-shan-p-591258244)
- **Adam Nahan(Adam P)** - [GitHub](https://github.com/Sha-330) | [LinkedIn](https://www.linkedin.com/in/adam-nahan-34a95524a?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BCLRzX0qSRBC%2FrcGGVwgkQw%3D%3D)
