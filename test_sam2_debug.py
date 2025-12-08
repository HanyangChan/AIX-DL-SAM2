print("Starting script...")
import os
print("Imported os")
import sys
print("Imported sys")

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))
print("Added backend to path")

try:
    import torch
    print("Imported torch")
except Exception as e:
    print(f"Failed to import torch: {e}")

try:
    from sam2_utils import load_sam2_model
    print("Imported sam2_utils")
except Exception as e:
    print(f"Failed to import sam2_utils: {e}")
    import traceback
    traceback.print_exc()

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
        print(f"Using device: {device}")
        model = load_sam2_model(sam2_cfg, sam2_ckpt, device=device)
        if model is not None:
            print("SUCCESS: SAM2 loaded successfully")
        else:
            print("FAIL: Model returned None")
    except Exception as e:
        print(f"FAIL: Exception during loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sam2_loading()
