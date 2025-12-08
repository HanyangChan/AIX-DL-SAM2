import sys
import os
import traceback

# Add backend to path
backend_path = os.path.join(os.getcwd(), 'backend')
sys.path.append(backend_path)

print(f"Checking syntax of backend/main.py...")
try:
    import main
    print("Syntax OK")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    traceback.print_exc()
except ImportError as e:
    print(f"ImportError (expected if dependencies missing): {e}")
except Exception as e:
    print(f"Other Error: {e}")
    traceback.print_exc()
