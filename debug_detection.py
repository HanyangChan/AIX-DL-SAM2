import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from sam2_utils import load_sam2_model, run_sam2_inference

import matplotlib.pyplot as plt

def debug_inference(image_path):
    print(f"Debugging image: {image_path}")
    
    # Load Model
    base_dir = os.path.join(os.getcwd(), 'backend')
    sam2_ckpt = os.path.join(base_dir, "sam2.1_hiera_tiny.pt")
    sam2_cfg = os.path.join(base_dir, "sam2.1_hiera_t.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading SAM2...")
    model = load_sam2_model(sam2_cfg, sam2_ckpt, device=device)
    
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Generate Grid (8x8 as per main.py)
    grid_x = np.linspace(0, w, 8, endpoint=False).astype(int)[1:]
    grid_y = np.linspace(0, h, 8, endpoint=False).astype(int)[1:]
    points = []
    for x in grid_x:
        for y in grid_y:
            points.append([x, y])
            
    print(f"Generated {len(points)} grid points")
    
    # Run Inference
    all_candidates = []
    
    for i, pt in enumerate(points):
        point_coords = np.array([pt])
        point_labels = np.array([1])
        
        _, sam_masks, scores = run_sam2_inference(
            model, 
            image_path, 
            points=point_coords, 
            labels=point_labels
        )
        
        best_idx = np.argmax(scores)
        best_mask = sam_masks[best_idx]
        best_score = scores[best_idx]
        
        if best_score < 0.3: # Threshold from main.py
            continue
            
        all_candidates.append((best_score, best_mask, pt))

    # Sort
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    print(f"\nCandidates passing score threshold: {len(all_candidates)}")
    
    # Visualization using OpenCV
    vis_img = img.copy()
    
    # NMS Simulation
    detected_masks = []
    
    # Generate random colors
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(all_candidates), 3), dtype=np.uint8)
    
    for idx, (score, mask, pt) in enumerate(all_candidates):
        mask_binary = (mask > 0.0).astype(np.uint8)
        area_mask_new = mask_binary.sum()
        
        # Draw point
        cv2.drawMarker(vis_img, (pt[0], pt[1]), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        
        print(f"\nCandidate {idx} from {pt}: Score {score:.3f}, Area {area_mask_new}")
        
        is_duplicate = False
        for i, existing in enumerate(detected_masks):
            existing_mask = existing['mask']
            
            # Mask IoU
            intersection = np.logical_and(mask_binary, existing_mask).sum()
            union = area_mask_new + existing['mask_area'] - intersection
            mask_iou = intersection / union if union > 0 else 0
            
            # Mask Containment
            mask_overlap_1 = intersection / area_mask_new if area_mask_new > 0 else 0
            mask_overlap_2 = intersection / existing['mask_area'] if existing['mask_area'] > 0 else 0
            
            print(f"  vs Existing {i}: MaskIoU {mask_iou:.2f}, Overlap1 {mask_overlap_1:.2f}, Overlap2 {mask_overlap_2:.2f}")
            
            if (mask_iou > 0.5) or (mask_overlap_1 > 0.8) or (mask_overlap_2 > 0.8):
                print("  -> SUPPRESSED")
                is_duplicate = True
                break
        
        if not is_duplicate:
            print("  -> ACCEPTED")
            detected_masks.append({
                'mask': mask_binary,
                'mask_area': area_mask_new
            })
            
            # Overlay mask
            color = colors[idx].tolist()
            
            # Find contours to draw outline
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, color, 2)
            
            # Add label
            y_indices, x_indices = np.where(mask_binary > 0)
            if len(y_indices) > 0:
                y_min, x_min = np.min(y_indices), np.min(x_indices)
                cv2.putText(vis_img, f"{idx}: {score:.2f}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = "debug_output.png"
    cv2.imwrite(output_path, vis_img)
    print(f"\nDebug image saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            img_path = sys.argv[1]
        else:
            # Try to find a default image
            img_path = "test data/test_image.jpg" # Example default
            if not os.path.exists(img_path):
                 # Try to find any png/jpg in current dir
                 for f in os.listdir("."):
                     if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                         img_path = f
                         break
            
        print(f"Using image: {img_path}")
        if not os.path.exists(img_path):
            print(f"Error: Image file not found at {img_path}")
            print("Usage: python debug_detection.py <path_to_image>")
        else:
            debug_inference(img_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
