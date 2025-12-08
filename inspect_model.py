import torch
import os
import sys

# Use absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "backend", "best_model.pth")

print(f"Checking model at: {model_path}")

if not os.path.exists(model_path):
    print(f"File not found: {model_path}")
else:
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        print(f"Loaded state_dict with {len(state_dict)} keys.")
        print("First 10 keys:")
        for i, key in enumerate(state_dict.keys()):
            if i >= 10: break
            print(f"  {key}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
