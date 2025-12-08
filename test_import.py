import sys
import os

try:
    with open("import_test.log", "w") as f:
        f.write("Starting import test...\n")
        
        import torch
        f.write(f"Torch version: {torch.__version__}\n")
        
        sys.path.append(os.path.join(os.getcwd(), 'backend'))
        f.write("Added backend to path\n")
        
        import sam2_utils
        f.write("Imported sam2_utils\n")
        
        from sam2_utils import load_sam2_model
        f.write("Imported load_sam2_model\n")
        
        f.write("Success!\n")
except Exception as e:
    with open("import_error.log", "w") as f:
        f.write(f"Error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
