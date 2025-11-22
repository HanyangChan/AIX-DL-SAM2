import cv2
import numpy as np
import base64

# Placeholder for SAM2 model loading
# In a real implementation, we would load the model here
# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def segment_image(image_path):
    """
    Segments the image using SAM2 (mocked for now).
    Returns a list of segments.
    """
    print(f"Segmenting image: {image_path}")
    
    # Dynamic implementation using Color Thresholding
    # This allows the segmentation to "react" to the actual image content
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    segments = []
    
    # Helper to process mask and add segment
    def add_segment(mask, label, id_val):
        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find largest contour to avoid noise
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
            
        # Sort by area and take top 1 or 2
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for i, cnt in enumerate(contours[:2]): # Take top 2 blobs max
            if cv2.contourArea(cnt) < (h*w*0.01): # Ignore very small blobs
                continue
                
            # Create clean mask for this blob
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            
            _, buffer = cv2.imencode('.png', blob_mask)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            segments.append({
                "id": f"{id_val}_{i}",
                "label": label,
                "mask": blob_mask,
                "mask_base64": mask_b64,
                "bbox": [x, y, x+bw, y+bh],
                "contours": cnt.reshape(-1, 2).tolist()
            })

    # 1. Meat (Reddish)
    # Red wraps around 180 in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_meat = cv2.bitwise_or(mask_red1, mask_red2)
    add_segment(mask_meat, "Meat", 1)

    # 2. Vegetable (Greenish)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_veg = cv2.inRange(img_hsv, lower_green, upper_green)
    add_segment(mask_veg, "Vegetable", 2)

    # 3. Mushroom (Brown/Dark/Yellowish)
    # Brown is tricky, often orange/yellow with low brightness
    lower_brown = np.array([10, 20, 20])
    upper_brown = np.array([30, 255, 200])
    mask_mush = cv2.inRange(img_hsv, lower_brown, upper_brown)
    add_segment(mask_mush, "Mushroom", 3)
    
    return segments
