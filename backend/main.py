from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import io

from model.inference import load_model, predict


MODEL_PATH = "edgecnn_v1.3.pth"
DEVICE = "cpu"

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float



app = FastAPI(title="EdgeCNN CIFAR-10 Image Classifier API")

# Load model at starup
model = load_model(model_path=MODEL_PATH, device=DEVICE)



# --- Root
@app.get("/")
def root():
    return {"message": "EdgeCNN API is running."}


# --- Health check 
@app.get("/health")
def health_check():
    '''
    Simple health check to confirm service is up
    '''
    return {
        "status": "ok", 
        "model_status": "loaded" if model is not None else "not loaded"
        }


# --- Prediction 
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile= File(...)): 
    '''
    Upload an image and get a CIFAR-10 class prediction w/ confidence from EdgeCNN
    '''
    try: 
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        pred_class, confidence = predict(model, image)

        return {
            "class_name": pred_class, 
            "confidence": confidence
            }
    
    except Exception as e: 
        raise HTTPException(status_code=400, detail=f"error: {str(e)}", )








  