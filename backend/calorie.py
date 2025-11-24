import numpy as np

def estimate_calories(segments, image_path):
    """
    Estimates calories based on segmentation masks.
    """
    # Calorie Density (kcal per full image coverage)
    # This assumes if the ENTIRE image was this food, it would be X calories.
    # This is a safer heuristic than raw pixels since image resolution varies.
    # Assumptions: Standard close-up food shot.
    MAX_CALORIES_PER_SCREEN = {
        "Meat": 2000.0,
        "Vegetable": 400.0,
        "Mushroom": 200.0,
        "Pizza": 2500.0,
        "Burger": 2200.0,
        "Rice": 1200.0,
        "Noodle": 1100.0,
        "Salad": 300.0,
        "Dessert": 3000.0,
        "Fruit": 500.0,
        "Soup": 600.0,
        "default": 600.0
    }

    # Get image dimensions for ratio calculation
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        total_area = 1000 * 1000 # Fallback
    else:
        h, w = img.shape[:2]
        total_area = h * w

    total_calories = 0
    food_items = []

    for seg in segments:
        # Heuristic:
        # 1. Calculate area in pixels
        mask = seg["mask"]
        area_pixels = np.sum(mask > 0)
        
        # 2. Calculate ratio of image covered
        ratio = area_pixels / total_area
        
        label = seg.get("label", "default")
        max_cals = MAX_CALORIES_PER_SCREEN.get(label, MAX_CALORIES_PER_SCREEN["default"])
        
        # 3. Estimate
        estimated_cals = ratio * max_cals
        
        total_calories += estimated_cals
        
        food_items.append({
            "label": label,
            "calories": round(estimated_cals, 1),
            "mask_base64": seg["mask_base64"],
            "bbox": seg.get("bbox", []),
            "contours": seg.get("contours", [])
        })
        
    return {
        "calories": round(total_calories, 1),
        "segmentation_mask": segments[0]["mask_base64"] if segments else "", # Return first mask as main for now, or we could combine them
        "food_items": food_items
    }
