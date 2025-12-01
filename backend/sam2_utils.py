import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except Exception as e:
    SAM2_AVAILABLE = False
    print(f"Warning: SAM2 module failed to load ({e}). Segmentation will use fallback.")

def load_sam2_model(config_path, checkpoint_path, device="cuda"):
    """
    Loads the SAM2 model.
    """
    if not SAM2_AVAILABLE:
        raise ImportError("SAM2 module is not installed.")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    print(f"Loading SAM2 model from {checkpoint_path}...")
    sam2_model = build_sam2(config_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def draw_mask(mask, ax, color=None, borders=True):
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(cnt, epsilon=0.01 * cv2.arcLength(cnt, True), closed=True) for cnt in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def draw_points(coords, labels, ax, marker_size=375):
    pos = coords[labels == 1]
    neg = coords[labels == 0]
    ax.scatter(pos[:, 0], pos[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg[:, 0], neg[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def draw_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def visualize_masks(image, masks, scores, point_coords=None, point_labels=None, box_coords=None, borders=True, save_path=None):
    """
    Visualizes the masks and optionally saves the result.
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        draw_mask(mask, ax, borders=borders)
        if point_coords is not None and point_labels is not None:
            draw_points(point_coords, point_labels, ax)
        if box_coords is not None:
            draw_box(box_coords, ax)
        title = f"Mask {i+1}, Score: {score:.3f}"
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.close(fig)

def run_sam2_inference(predictor, image_path, box=None, points=None, labels=None):
    """
    Runs SAM2 inference on a single image.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    predictor.set_image(image_np)
    
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=box,
        multimask_output=False,
    )
    
    return image_np, masks, scores
