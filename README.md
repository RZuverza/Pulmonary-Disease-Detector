# Pulmonary Disease Detector



## Application Preview

![Application Screenshot](Pulmonary%20Disease%20Detector.png)

Deep Learning application for chest X-ray classification using TensorFlow and Streamlit.


## Overview

This project uses a trained TensorFlow/Keras model to classify chest X-ray images into four categories:

- COVID
- Lung Opacity
- Normal
- Pneumonia

The application includes an interactive Streamlit interface that allows users to upload a chest X-ray image and obtain predictions with confidence scores.

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Project Structure

```text
app.py
modelo_covid_v2.keras
requirements.txt
covid_v2.ipynb
dataset/
README.md
```

## Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- Chest X-ray image upload
- Real-time disease classification
- Confidence score visualization
- Probability distribution chart
- Simple web interface

## Author

Ricardo García Zuverza.
