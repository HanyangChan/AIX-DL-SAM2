from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
import os
from segmentation import segment_image
from calorie import estimate_calories

app = FastAPI(title="Food Calorie Estimator")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    calories: float
    segmentation_mask: str # Base64 encoded mask or URL
    food_items: list[dict]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. Segment Image
        # Returns masks and class names/ids
        segments = segment_image(temp_file)
        
        # 2. Estimate Calories
        # Calculates calories based on segments
        result = estimate_calories(segments, temp_file)
        
        # Cleanup
        os.remove(temp_file)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
