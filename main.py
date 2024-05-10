import json
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
from lime import lime_image
from matplotlib.pyplot import subplots
from skimage.segmentation import mark_boundaries
import logger as log
import base64

modelFilePath = './agroTechModel.h5'
diseaseFilePath = './diseaseInfo.json'  
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
explainer = lime_image.LimeImageExplainer()
explanation = None

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

def load_disease_info(file_path):
    with open(file_path, 'r') as file:
        disease_data = json.load(file)
    return disease_data


disease_data = load_disease_info(diseaseFilePath)

@app.post('/predict', response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    global explanation
    contents = await file.read()
    log.logging.info('Here')
    image = Image.open(io.BytesIO(contents)) 
    
    image = np.array(image) 
    
    preprocessed_image = preprocess_image(image)
    
    explanation = explainer.explain_instance(preprocessed_image[0].astype('double'), model.predict,
                                         top_labels=3, hide_color=0, num_samples=100)
    
    prediction = predict_disease(preprocessed_image)
    
    
    pred = str(prediction[0])
    
    if pred in disease_data:
        return {'prediction' : json.dumps(disease_data[pred])}
    else:
        return {'prediction' : 'Unknown Disease'}
    

@app.post('/getExplaination')
async def getExplaination():
    global explanation  # Use the global explanation variable
    if explanation:
        temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

        fig, (ax1, ax2) = subplots(1, 2, figsize=(15,15))
        ax1.imshow(mark_boundaries(temp_1, mask_1))
        ax2.imshow(mark_boundaries(temp_2, mask_2))
        ax1.axis('off')
        ax2.axis('off')

        # Convert the plot to bytes
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        img_base64 = img_bytes.getvalue()

         # Encode the image bytes as Base64
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        return {'explanation_image': img_base64}
    else:
        return {'message': 'Explanation not available'}
    
@app.get("/")
def index():
    return {'message' : 'Hello World'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, debug=True)

    
#python -m uvicorn main:app --reload
# web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app


#python -m uvicorn main:app --host 0.0.0.0 --port 8000


# Local host URL
# http://127.0.0.1:8000/

# Public URL
# http://192.168.10.40:8000/predict

