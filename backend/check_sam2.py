import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Current Directory: {os.getcwd()}")

print("Attempting to import sam2...")
try:
    import sam2
    print("SUCCESS: sam2 imported correctly!")
    print(f"sam2 location: {sam2.__file__}")
except ImportError as e:
    print("\n[FAILURE] Could not import sam2.")
    print(f"Error Message: {e}")
    print("\nPossible solutions:")
    print("1. Run 'pip install -r requirements.txt' again.")
    print("2. Ensure you have git installed.")
    print("3. Try 'pip install git+https://github.com/facebookresearch/sam2.git' manually.")
except Exception as e:
    print(f"\n[FAILURE] An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()

input("\nPress Enter to exit...")
