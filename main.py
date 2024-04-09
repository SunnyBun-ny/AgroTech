import boto3
import os
import uvicorn
import cv2
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
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

# AWS credentials from Heroku config vars
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')  # Update with your S3 bucket name
MODEL_KEY = 'agroTechModel.h5'     # Update with the key of your h5 model file in the bucket

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# Function to load h5 model from S3
def load_h5_model_from_s3(bucket_name, key):
    response = s3.get_object(Bucket=bucket_name, Key=key)
    model_bytes = response['Body'].read()
    model = load_model(BytesIO(model_bytes))
    return model

# Load the model from S3
model = load_h5_model_from_s3(BUCKET_NAME, MODEL_KEY)





app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


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

@app.post('/predict', response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)) 
    image = np.array(image) 
    preprocessed_image = preprocess_image(image)
    prediction = predict_disease(preprocessed_image)
    
    pred = int(prediction[0])

    if pred in disease_mapping:
        return {'prediction' :  disease_mapping[pred]}
    else:
        return {'prediction' : 'Unknown Disease'}

@app.get("/")
def index():
    return {'message' : 'Hello World'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#python -m uvicorn main:app --reload