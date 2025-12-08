import numpy as np



# Average calories per serving/item (approximate)
AVERAGE_CALORIES = {
    "apple_pie": 300,
    "baby_back_ribs": 500, # Not in list but good to have
    "baklava": 330,
    "beef_carpaccio": 150,
    "beef_tartare": 200,
    "beet_salad": 150,
    "beignets": 250,
    "bibimbap": 600,
    "bread_pudding": 350,
    "breakfast_burrito": 400,
    "bruschetta": 100,
    "burger": 500,
    "caesar_salad": 350,
    "cannoli": 250,
    "caprese_salad": 200,
    "carrot_cake": 400,
    "ceviche": 150,
    "cheesecake": 400,
    "cheese_plate": 400,
    "chicken_curry": 450,
    "chicken_quesadilla": 450,
    "chicken_wings": 100, # Per wing? Let's say serving of 3-4
    "chocolate_cake": 450,
    "chocolate_mousse": 300,
    "churros": 250,
    "clam_chowder": 300,
    "club_sandwich": 450,
    "crab_cakes": 250,
    "creme_brulee": 350,
    "croque_madame": 500,
    "cup_cakes": 250,
    "deviled_eggs": 100,
    "donuts": 250,
    "dumplings": 50, # Per dumpling
    "edamame": 100,
    "eggs_benedict": 400,
    "escargots": 150,
    "falafel": 300,
    "filet_mignon": 400,
    "fish_and_chips": 600,
    "foie_gras": 300,
    "french_fries": 350,
    "french_onion_soup": 250,
    "french_toast": 400,
    "fried_calamari": 350,
    "fried_rice": 400,
    "frozen_yogurt": 200,
    "garlic_bread": 200,
    "gnocchi": 350,
    "greek_salad": 200,
    "grilled_cheese_sandwich": 400,
    "grilled_salmon": 350,
    "guacamole": 200,
    "gyoza": 50,
    "hamburger": 500,
    "hot_and_sour_soup": 150,
    "hot_dog": 300,
    "huevos_rancheros": 450,
    "hummus": 200,
    "ice_cream": 250,
    "lasagna": 500,
    "lobster_bisque": 300,
    "lobster_roll_sandwich": 450,
    "macaroni_and_cheese": 400,
    "macarons": 100,
    "miso_soup": 80,
    "mussels": 250,
    "nachos": 500,
    "omelette": 300,
    "onion_rings": 350,
    "oysters": 100,
    "pad_thai": 500,
    "paella": 500,
    "pancakes": 350,
    "panna_cotta": 300,
    "peking_duck": 500,
    "pho": 350,
    "pizza": 300, # Per slice
    "pork_chop": 400,
    "poutine": 500,
    "prime_rib": 600,
    "pulled_pork_sandwich": 500,
    "ramen": 500,
    "ravioli": 350,
    "red_velvet_cake": 400,
    "risotto": 400,
    "samosa": 150,
    "sashimi": 200,
    "scallops": 200,
    "seaweed_salad": 100,
    "shrimp_and_grits": 450,
    "spaghetti_bolognese": 450,
    "spaghetti_carbonara": 500,
    "spring_rolls": 150,
    "steak": 500,
    "strawberry_shortcake": 300,
    "sushi": 300, # Per roll/serving
    "tacos": 200, # Per taco
    "takoyaki": 200,
    "tiramisu": 350,
    "tuna_tartare": 200,
    "waffles": 350,
    # Specific classes from Kaggle dataset
    "baked_potato": 300,
    "crispy_chicken": 400,
    "donut": 250,
    "fries": 350,
    "sandwich": 400,
    "taco": 200,
    "taquito": 200,
    "butter_naan": 250,
    "chai": 100,
    "chapati": 100,
    "chole_bhature": 450,
    "dal_makhani": 300,
    "dhokla": 150,
    "idli": 50,
    "jalebi": 200,
    "kaathi_rolls": 350,
    "kadai_paneer": 350,
    "kulfi": 200,
    "masala_dosa": 350,
    "momos": 200,
    "paani_puri": 150,
    "pakode": 250,
    "pav_bhaji": 400,
    "default": 300
}

def estimate_calories(segments, image_path):
    """
    Estimates calories based on fixed average values per class.
    """
    # We don't need image_path anymore for weight estimation, 
    # but we keep the signature compatible or use it if we want to log something.
    
    total_calories = 0
    food_items = []

    for seg in segments:
        label = seg.get("label", "default")
        
        # Normalize label
        normalized_label = label.lower().replace(" ", "_")
        
        # Lookup calories
        # Try exact match, then normalized, then default
        calories = AVERAGE_CALORIES.get(label, AVERAGE_CALORIES.get(normalized_label, AVERAGE_CALORIES["default"]))
        
        total_calories += calories
        
        food_items.append({
            "label": label,
            "calories": calories,
            "weight_g": 0, # Dummy value or remove if frontend doesn't need it
            "mask_base64": seg.get("mask_base64", ""),
            "bbox": seg.get("bbox", []),
            "contours": seg.get("contours", [])
        })
        
    return {
        "calories": round(total_calories, 1),
        "segmentation_mask": segments[0]["mask_base64"] if segments and "mask_base64" in segments[0] else "",
        "food_items": food_items
    }
