import uvicorn
import cv2
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
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
app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load_model(modelFilePath)


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
    image = Image.open(io.BytesIO(contents)) 
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

    
#python -m uvicorn main:app --reload