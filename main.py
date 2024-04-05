import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from typing import List
import io
from PIL import Image


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

def predictDisease(image):
    result = model.predict(image) # predict diseased palnt or not
    pred = np.argmax(result, axis=1)
    return pred

def preprocess_image(image):
    test_image = cv2.resize(image, (256,256)) # load image
    test_image = img_to_array(test_image)/255 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0) #
    return test_image


@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    image = Image.open(io.BytesIO(await files[0].read()))
    prediction = predictDisease(image)
    return {"prediction": prediction}
