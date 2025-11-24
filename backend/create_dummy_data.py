import os
import numpy as np
import cv2

def create_dummy_data():
    base_dir = 'backend/dummy_dataset'
    classes = ['classA', 'classB']
    sets = ['train', 'test']
    
    for s in sets:
        for c in classes:
            dir_path = os.path.join(base_dir, s, c)
            os.makedirs(dir_path, exist_ok=True)
            
            # Create 5 dummy images per class
            for i in range(5):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(dir_path, f'img_{i}.png'), img)
                
    print(f"Created dummy dataset at {base_dir}")

if __name__ == "__main__":
    create_dummy_data()
