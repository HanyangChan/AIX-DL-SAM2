import os
import torch
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from sam2_utils import load_sam2_model

def test_sam2_loading():
    base_dir = os.path.join(os.getcwd(), 'backend')
    sam2_ckpt = os.path.join(base_dir, "sam2.1_hiera_tiny.pt")
    sam2_cfg = os.path.join(base_dir, "sam2.1_hiera_t.yaml")
    
    print(f"Testing SAM2 loading...")
    print(f"Checkpoint: {sam2_ckpt}")
    print(f"Config: {sam2_cfg}")
    
    if not os.path.exists(sam2_ckpt):
        print("FAIL: Checkpoint not found")
        return
    if not os.path.exists(sam2_cfg):
        print("FAIL: Config not found")
        return
        
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_sam2_model(sam2_cfg, sam2_ckpt, device=device)
        if model is not None:
            print("SUCCESS: SAM2 loaded successfully")
        else:
            print("FAIL: Model returned None")
    except Exception as e:
        print(f"FAIL: Exception during loading: {e}")

if __name__ == "__main__":
    test_sam2_loading()
