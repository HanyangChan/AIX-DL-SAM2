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
import numpy as np

def read_image_safe(path):
    """
    Reads an image from a path, handling Unicode characters correctly on Windows.
    """
    try:
        # Read file as byte array
        with open(path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            # Decode image
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(base_dir, "model_metadata.json")
        model_path = os.path.join(base_dir, "best_model.pth")

        if os.path.exists(metadata_path) and os.path.exists(model_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                models["class_names"] = meta["class_names"]
            
            print(f"Loading classifier with classes: {models['class_names']}")
            model = initialize_model(len(models["class_names"]))
            model.load_state_dict(torch.load(model_path, map_location=models["device"]))
            model.eval()
            models["classifier"] = model
            print("Classifier loaded successfully.")
        else:
            print(f"Model files not found at {metadata_path} or {model_path}. Using dummy classes.")
            models["class_names"] = ["classA", "classB"]
            
    except Exception as e:
        print(f"Failed to load classifier: {e}")
        models["class_names"] = ["classA", "classB"]

    # Load SAM2
    try:
        sam2_ckpt = "backend/sam2.1_hiera_tiny.pt"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml" 
        # Note: sam2 library usually expects config relative to its package or absolute path.
        # If using sam2.build_sam.build_sam2, we might need just "sam2.1_hiera_t.yaml" if it's in the default configs.
        # Let's try the simple name first, if it fails we might need to find where the config is.
        # Actually, for sam2.1, the config name is likely "configs/sam2.1/sam2.1_hiera_t.yaml" if using the repo,
        # but the installed package might handle "sam2.1_hiera_t.yaml".
        # Let's stick to a safe guess or try to find it. 
        # Reverting to simple name as per standard usage in tutorials.
        sam2_cfg = "sam2.1_hiera_t.yaml"
        
        if os.path.exists(sam2_ckpt):
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

def segment_with_opencv(image_path):
    """
    Segments objects using OpenCV contours (fallback for missing SAM2).
    Assumes objects are on a lighter background or distinct.
    """
    img = read_image_safe(image_path)
    if img is None:
        return []
        
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding
    # Try Otsu's binarization
    # Invert if background is white (common in food photos)
    # We assume food is darker than white background
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    masks = []
    for cnt in contours:
        # Filter small noise
        area = cv2.contourArea(cnt)
        if area < 1000: # Minimum area threshold
            continue
            
        # Create mask for this contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, -1)
        masks.append(mask)
        
    # If no contours found, try standard thresholding (maybe dark background?)
    if not masks:
         _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         for cnt in contours:
            if cv2.contourArea(cnt) < 1000: continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, -1)
            masks.append(mask)
            
    return masks

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Segmentation (SAM2) - Grid Prompting
        masks = []
        
        # Read image safely
        img = read_image_safe(temp_file)
        if img is None:
             raise ValueError(f"Failed to read image: {temp_file}")
        h, w = img.shape[:2]

        if models["sam2"]:
            # Generate grid points (3x3 grid)
            grid_x = [w // 4, w // 2, 3 * w // 4]
            grid_y = [h // 4, h // 2, 3 * h // 4]
            points = []
            for x in grid_x:
                for y in grid_y:
                    points.append([x, y])
            
            detected_masks = []
            
            # Run inference for each point
            print(f"Running SAM2 on {len(points)} points...")
            for i, pt in enumerate(points):
                point_coords = np.array([pt])
                point_labels = np.array([1])
                
                _, sam_masks, scores = run_sam2_inference(
                    models["sam2"], 
                    temp_file, 
                    points=point_coords, 
                    labels=point_labels
                )
                
                # sam_masks is (3, H, W), scores is (3,)
                # Take best mask
                best_idx = np.argmax(scores)
                best_mask = sam_masks[best_idx]
                best_score = scores[best_idx]
                
                print(f"Point {pt}: Score {best_score:.3f}")

                # Filter weak predictions
                if best_score < 0.80: # Threshold
                    print(f"  -> Skipped (Low Score)")
                    continue
                    
                # Check IoU with existing masks to avoid duplicates
                is_duplicate = False
                for existing_mask in detected_masks:
                    # Calculate IoU
                    intersection = np.logical_and(best_mask, existing_mask).sum()
                    union = np.logical_or(best_mask, existing_mask).sum()
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.5: # High overlap means same object
                        is_duplicate = True
                        print(f"  -> Skipped (Duplicate, IoU: {iou:.2f})")
                        break
                
                if not is_duplicate:
                    print(f"  -> Added new mask")
                    detected_masks.append(best_mask)
            
            print(f"Total masks found: {len(detected_masks)}")

            # If no masks found (e.g. background points), fallback to center or dummy
            if not detected_masks:
                 print("No masks found, falling back to dummy.")
                 # Fallback to center
                 dummy_mask = np.zeros((h, w), dtype=np.uint8)
                 cv2.circle(dummy_mask, (w//2, h//2), min(h, w)//3, 1, -1)
                 detected_masks.append(dummy_mask)
                 
            masks = [{"mask": m} for m in detected_masks]

        else:
            # Fallback: OpenCV Segmentation
            print("SAM2 not available. Using OpenCV fallback.")
            cv_masks = segment_with_opencv(temp_file)
            if cv_masks:
                masks = [{"mask": m} for m in cv_masks]
            else:
                # Dummy segmentation if OpenCV fails
                dummy_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(dummy_mask, (w//2, h//2), min(h, w)//3, 1, -1)
                masks.append({"mask": dummy_mask})

        # 2. Classification per Mask (Crop & Classify)
        if models["classifier"]:
            # Pre-load full image tensor for cropping? 
            # Actually easier to crop numpy image then transform.
            pil_image = Image.open(temp_file).convert("RGB")
            
            for m in masks:
                mask = m["mask"]
                
                # Find bounding box of mask
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0:
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    
                    # Add padding
                    pad = 10
                    y_min = max(0, y_min - pad)
                    y_max = min(h, y_max + pad)
                    x_min = max(0, x_min - pad)
                    x_max = min(w, x_max + pad)
                    
                    # Crop
                    crop = pil_image.crop((x_min, y_min, x_max, y_max))
                    
                    # Transform for model
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform(crop).unsqueeze(0).to(models["device"])
                    
                    # Classify
                    with torch.no_grad():
                        outputs = models["classifier"](input_tensor)
                        _, preds = torch.max(outputs, 1)
                        predicted_class = models["class_names"][preds.item()]
                        m["label"] = predicted_class
                        m["bbox"] = [int(x_min), int(y_min), int(x_max), int(y_max)]
                else:
                    m["label"] = "Unknown"
                    m["bbox"] = []
        else:
             for m in masks:
                 m["label"] = "Unknown"

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
