from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
import shutil
import os
import json
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import base64
import io

# Import project modules
from calorie import estimate_calories
from train_classifier import initialize_model
from sam2_utils import load_sam2_model, run_sam2_inference

# Global variables for models
models = {
    "classifier": None,
    "sam2": None,
    "class_names": [],
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Classifier
    try:
        if os.path.exists("backend/model_metadata.json") and os.path.exists("backend/best_model.pth"):
            with open("backend/model_metadata.json", "r") as f:
                meta = json.load(f)
                models["class_names"] = meta["class_names"]
            
            print(f"Loading classifier with classes: {models['class_names']}")
            model = initialize_model(len(models["class_names"]))
            model.load_state_dict(torch.load("backend/best_model.pth", map_location=models["device"]))
            model.eval()
            models["classifier"] = model
            print("Classifier loaded successfully.")
        else:
            print("Warning: Classifier model or metadata not found. Using dummy classification.")
    except Exception as e:
        print(f"Error loading classifier: {e}")

    # Load SAM2
    try:
        sam2_ckpt = "backend/sam2_hiera_large.pt"
        sam2_cfg = "sam2_hiera_l.yaml" # Assumes this config is in the path or handled by sam2 library
        # Note: sam2 config loading might be tricky depending on where the library looks. 
        # For now, we assume the user has the config or we use the default if possible.
        # If sam2_utils handles it, great.
        
        if os.path.exists(sam2_ckpt):
             # We might need to adjust config path logic
             models["sam2"] = load_sam2_model(sam2_cfg, sam2_ckpt, device=models["device"])
             print("SAM2 loaded successfully.")
        else:
             print("Warning: SAM2 checkpoint not found. Using dummy segmentation.")
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        
    yield
    
    # Cleanup
    models.clear()

app = FastAPI(title="Food Calorie Estimator", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    calories: float
    segmentation_mask: str
    food_items: list[dict]

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def encode_mask(mask):
    # Convert boolean/float mask to uint8 image
    mask_uint8 = (mask * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', mask_uint8)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "classifier_loaded": models["classifier"] is not None,
        "sam2_loaded": models["sam2"] is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Classification
        predicted_class = "Unknown"
        if models["classifier"]:
            input_tensor = preprocess_image(temp_file).to(models["device"])
            with torch.no_grad():
                outputs = models["classifier"](input_tensor)
                _, preds = torch.max(outputs, 1)
                predicted_class = models["class_names"][preds.item()]
        
        # 2. Segmentation (SAM2)
        masks = []
        if models["sam2"]:
            # Use center point as prompt
            img = cv2.imread(temp_file)
            h, w = img.shape[:2]
            center_point = np.array([[w//2, h//2]])
            point_labels = np.array([1]) # 1 = foreground
            
            _, sam_masks, _ = run_sam2_inference(
                models["sam2"], 
                temp_file, 
                points=center_point, 
                labels=point_labels
            )
            # sam_masks is (N, H, W). We take the best one (usually index 0 for single object)
            best_mask = sam_masks[0]
            masks.append({"mask": best_mask, "label": predicted_class})
        else:
            # Dummy segmentation
            img = cv2.imread(temp_file)
            h, w = img.shape[:2]
            dummy_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(dummy_mask, (w//2, h//2), min(h, w)//3, 1, -1)
            masks.append({"mask": dummy_mask, "label": predicted_class})

        # 3. Calorie Estimation
        # We need to pass the encoded mask for the frontend
        for m in masks:
            m["mask_base64"] = encode_mask(m["mask"])
            
        result = estimate_calories(masks, temp_file)
        
        os.remove(temp_file)
        return result
        
    except Exception as e:
        if os.path.exists(f"temp_{file.filename}"):
            os.remove(f"temp_{file.filename}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
