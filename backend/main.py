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
        # Use absolute paths based on the current file's location
        sam2_ckpt = os.path.join(base_dir, "sam2.1_hiera_tiny.pt")
        
        # Check for config in the same directory first
        local_config = os.path.join(base_dir, "sam2.1_hiera_t.yaml")
        if os.path.exists(local_config):
            sam2_cfg = local_config
        else:
            # Fallback to simple name if installed in package
            sam2_cfg = "sam2.1_hiera_t.yaml"
        
        if os.path.exists(sam2_ckpt):
             print(f"Loading SAM2 from {sam2_ckpt} with config {sam2_cfg}...")
             models["sam2"] = load_sam2_model(sam2_cfg, sam2_ckpt, device=models["device"])
             print("SAM2 loaded successfully.")
        else:
             print(f"Warning: SAM2 checkpoint not found at {sam2_ckpt}. Using dummy segmentation.")
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
    # Debug: Start of predict
    try:
        with open("debug_start.log", "w") as f:
            f.write("Predict called\n")
            f.write(f"CWD: {os.getcwd()}\n")
    except:
        pass

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
            # Generate grid points (12x12 grid) - Ultra dense
            grid_x = np.linspace(0, w, 12, endpoint=False).astype(int)[1:] 
            grid_y = np.linspace(0, h, 12, endpoint=False).astype(int)[1:] 
            
            step_x = w // 12
            step_y = h // 12
            grid_x = [step_x * i for i in range(1, 12)]
            grid_y = [step_y * i for i in range(1, 12)]
            
            points = []
            for x in grid_x:
                for y in grid_y:
                    points.append([x, y])
            
            # Collect all candidates first
            all_candidates = [] # List of (score, mask) tuples
            
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
                
                # print(f"Point {pt}: Score {best_score:.3f}") # Reduce log spam

                # Filter weak predictions - Very low threshold
                if best_score < 0.10: 
                    # print(f"  -> Skipped (Low Score)")
                    continue
                
                all_candidates.append((best_score, best_mask))

            # --- Class-Aware NMS ---
            # 1. Sort by score descending
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # 2. Classify ALL candidates first (if classifier available)
            # We need to crop and classify each candidate mask
            candidate_dicts = []
            
            pil_image = Image.open(temp_file).convert("RGB")
            
            print(f"Classifying {len(all_candidates)} candidates for NMS...")
            
            for score, mask in all_candidates:
                mask_binary = (mask > 0.0).astype(np.uint8)
                
                # Calculate Box
                y_indices, x_indices = np.where(mask_binary > 0)
                if len(y_indices) == 0: continue
                
                y1, y2 = int(np.min(y_indices)), int(np.max(y_indices))
                x1, x2 = int(np.min(x_indices)), int(np.max(x_indices))
                box = [x1, y1, x2, y2]
                area_mask = int(mask_binary.sum())
                area_box = int((x2 - x1) * (y2 - y1))
                
                label = "Unknown"
                if models["classifier"]:
                    # Add padding
                    pad = 10
                    cy1 = max(0, y1 - pad)
                    cy2 = min(h, y2 + pad)
                    cx1 = max(0, x1 - pad)
                    cx2 = min(w, x2 + pad)
                    
                    crop = pil_image.crop((cx1, cy1, cx2, cy2))
                    
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform(crop).unsqueeze(0).to(models["device"])
                    
                    with torch.no_grad():
                        outputs = models["classifier"](input_tensor)
                        _, preds = torch.max(outputs, 1)
                        label = models["class_names"][preds.item()]
                
                # Filter out background masks (too large) or tiny noise (too small)
                image_area = h * w
                if area_box > image_area * 0.95:
                    continue
                if area_box < image_area * 0.005: # Filter < 0.5% area
                    continue

                candidate_dicts.append({
                    'score': score,
                    'mask': mask_binary,
                    'mask_area': area_mask,
                    'box': box,
                    'box_area': area_box,
                    'label': label
                })

            # --- Voting NMS Logic ---
            # 1. Cluster candidates based on spatial proximity/overlap
            clusters = []
            
            # Open debug log
            base_dir = os.path.dirname(os.path.abspath(__file__))
            debug_log_path = os.path.join(base_dir, "debug_nms.log")
            
            with open(debug_log_path, "w") as log_f:
                log_f.write(f"Processing {len(candidate_dicts)} candidates for Voting NMS...\n")
                
                for idx, curr in enumerate(candidate_dicts):
                    log_f.write(f"\nCandidate {idx}: Label '{curr['label']}', Score {curr['score']:.3f}, Box {curr['box']}\n")
                    
                    best_cluster_idx = -1
                    best_cluster_score = -1
                    
                    # Find best matching cluster
                    for c_idx, cluster in enumerate(clusters):
                        # Compare with cluster representative (first item, highest score)
                        existing = cluster['representative']
                        
                        # Calculate metrics
                        intersection = np.logical_and(curr['mask'], existing['mask']).sum()
                        union = curr['mask_area'] + existing['mask_area'] - intersection
                        mask_iou = intersection / union if union > 0 else 0
                        
                        ix1 = max(curr['box'][0], existing['box'][0])
                        iy1 = max(curr['box'][1], existing['box'][1])
                        ix2 = min(curr['box'][2], existing['box'][2])
                        iy2 = min(curr['box'][3], existing['box'][3])
                        box_iou = 0
                        if ix1 < ix2 and iy1 < iy2:
                            inter_box_area = (ix2 - ix1) * (iy2 - iy1)
                            union_box_area = curr['box_area'] + existing['box_area'] - inter_box_area
                            box_iou = inter_box_area / union_box_area if union_box_area > 0 else 0
                            
                        mask_overlap_1 = intersection / curr['mask_area'] if curr['mask_area'] > 0 else 0
                        mask_overlap_2 = intersection / existing['mask_area'] if existing['mask_area'] > 0 else 0

                        cx_curr = (curr['box'][0] + curr['box'][2]) / 2
                        cy_curr = (curr['box'][1] + curr['box'][3]) / 2
                        cx_exist = (existing['box'][0] + existing['box'][2]) / 2
                        cy_exist = (existing['box'][1] + existing['box'][3]) / 2
                        dist = ((cx_curr - cx_exist)**2 + (cy_curr - cy_exist)**2)**0.5
                        
                        w_exist = existing['box'][2] - existing['box'][0]
                        h_exist = existing['box'][3] - existing['box'][1]
                        diag_exist = (w_exist**2 + h_exist**2)**0.5
                        norm_dist = dist / diag_exist if diag_exist > 0 else 1.0
                        
                        area_ratio = curr['mask_area'] / existing['mask_area'] if existing['mask_area'] > 0 else 0
                        
                        log_f.write(f"  vs Cluster {c_idx} ({existing['label']}): MaskIoU {mask_iou:.2f}, BoxIoU {box_iou:.2f}, Dist {norm_dist:.2f}\n")

                        # Match Condition (Same as previous suppression logic)
                        is_match = False
                        if curr['label'] == existing['label']:
                            if (mask_iou > 0.1 or box_iou > 0.05 or mask_overlap_1 > 0.4 or mask_overlap_2 > 0.4 or norm_dist < 0.75):
                                is_match = True
                        else:
                            if (mask_iou > 0.5 or mask_overlap_1 > 0.8 or mask_overlap_2 > 0.8 or 
                                (norm_dist < 0.1 and area_ratio < 0.1) or 
                                (box_iou > 0.01 and area_ratio < 0.1) or 
                                (norm_dist < 0.5)):
                                is_match = True
                        
                        if is_match:
                            # If multiple clusters match, pick the one with highest score representative?
                            # For simplicity, pick the first strong match or merge?
                            # Let's just pick the first match for now.
                            best_cluster_idx = c_idx
                            break
                    
                    if best_cluster_idx != -1:
                        log_f.write(f"  -> Added to Cluster {best_cluster_idx}\n")
                        clusters[best_cluster_idx]['candidates'].append(curr)
                    else:
                        log_f.write(f"  -> New Cluster {len(clusters)}\n")
                        clusters.append({
                            'representative': curr,
                            'candidates': [curr]
                        })

                # 2. Resolve Clusters (Voting)
                detected_masks = []
                log_f.write(f"\nResolving {len(clusters)} clusters...\n")
                
                for c_idx, cluster in enumerate(clusters):
                    # Vote for label
                    label_scores = {}
                    for cand in cluster['candidates']:
                        lbl = cand['label']
                        label_scores[lbl] = label_scores.get(lbl, 0) + cand['score']
                    
                    # Winner
                    winner_label = max(label_scores, key=label_scores.get)
                    log_f.write(f"Cluster {c_idx}: Votes {label_scores} -> Winner: {winner_label}\n")
                    
                    # Merge masks of winner label
                    winner_candidates = [c for c in cluster['candidates'] if c['label'] == winner_label]
                    
                    # Base mask is the highest score one (representative of that class)
                    # Or union of all? Union is better for fragments.
                    merged_mask = np.zeros_like(winner_candidates[0]['mask'])
                    for wc in winner_candidates:
                        merged_mask = np.logical_or(merged_mask, wc['mask'])
                    
                    # Recalculate Box
                    y_indices, x_indices = np.where(merged_mask)
                    if len(y_indices) > 0:
                        y_min, y_max = y_indices.min(), y_indices.max()
                        x_min, x_max = x_indices.min(), x_indices.max()
                        merged_box = [int(x_min), int(y_min), int(x_max), int(y_max)]
                    else:
                        merged_box = winner_candidates[0]['box'] # Fallback
                        
                    detected_masks.append({
                        'mask': merged_mask,
                        'label': winner_label,
                        'box': merged_box,
                        'score': label_scores[winner_label] # Use total score or max score? Let's use total for sorting if needed
                    })

            print(f"Total masks found after Class-Aware NMS: {len(detected_masks)}")
            
            # Format results
            masks = []
            for d in detected_masks:
                masks.append({
                    "mask": d['mask'],
                    "label": d['label'],
                    "bbox": d['box']
                })
            
            with open("debug_progress.log", "a") as f:
                f.write("NMS and formatting done\n")

        else:
            # Fallback: OpenCV Segmentation
            print("SAM2 not available. Using OpenCV fallback.")
            cv_masks = segment_with_opencv(temp_file)
            if cv_masks:
                masks = [{"mask": m, "label": "Unknown", "bbox": []} for m in cv_masks]
            else:
                # Dummy segmentation if OpenCV fails
                dummy_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(dummy_mask, (w//2, h//2), min(h, w)//3, 1, -1)
                masks.append({"mask": dummy_mask, "label": "Unknown", "bbox": []})

        # 3. Calorie Estimation (Classification already done)
        # We need to pass the encoded mask for the frontend
        with open("debug_progress.log", "a") as f:
            f.write(f"Starting mask encoding for {len(masks)} masks\n")
            
        for i, m in enumerate(masks):
            try:
                m["mask_base64"] = encode_mask(m["mask"])
            except Exception as e:
                with open("debug_progress.log", "a") as f:
                    f.write(f"Error encoding mask {i}: {e}\n")
                raise e
                
        with open("debug_progress.log", "a") as f:
            f.write("Mask encoding done. Calling estimate_calories\n")
            
        result = estimate_calories(masks, temp_file)
        
        with open("debug_progress.log", "a") as f:
            f.write("estimate_calories done. Cleaning up.\n")
        
        os.remove(temp_file)
        return result
        
    except Exception as e:
        # Write crash log
        base_dir = os.path.dirname(os.path.abspath(__file__))
        crash_log_path = os.path.join(base_dir, "crash.log")
        with open(crash_log_path, "w") as f:
            f.write(f"Error: {str(e)}\n")
            import traceback
            traceback.print_exc(file=f)
            
        if os.path.exists(f"temp_{file.filename}"):
            os.remove(f"temp_{file.filename}")
        print(f"Error in predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
