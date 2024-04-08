import uvicorn
import cv2
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from typing import List
import io
from PIL import Image
from pydantic import BaseModel

disease_mapping = {
    0: "Canker Fruits Disease",
    1: "Caterpillar Forms Leaf Disease",
    2: "Faint Color Fruit Disease",
    3: "InitialBurn Leaf Disease",
    4: "Myrtle Rust Leaf disease",
    5: "Nutrition Deficiency Leaf disease",
    6: "SemiBurn Leaf Disease",
    7: "Crack Fruits Disease",
    8: "Diplodia Fruits Disease",
    9: "Fungi Fruits Disease",
    10: "Shrink Leaf Disease",
    11: "Uneven Size Fruit Disease"
}

modelFilePath = './agroTechModel.h5'
modelTestImage = './IMG_20221026_115327.jpg'
model = load_model(modelFilePath)
testImage = cv2.imread(modelTestImage)

class Prediction(BaseModel):
    prediction: str

def predict_disease(image):
    result = model.predict(image) # predict diseased palnt or not
    pred = np.argmax(result, axis=1)
    return pred

def preprocess_image(image):
    test_image = cv2.resize(image, (256,256))
    test_image = img_to_array(test_image)/255 
    test_image = np.expand_dims(test_image, axis = 0) 
    return test_image

def predict():
    preprocessed_image = preprocess_image(testImage)
    prediction = predict_disease(preprocessed_image)
    
    pred = int(prediction[0])

    if pred in disease_mapping:
        print( disease_mapping[pred])
    else:
        print("Unknown Disease")

    
#python -m uvicorn main:app --reload