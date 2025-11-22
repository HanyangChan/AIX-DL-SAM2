import requests
import cv2
import numpy as np

# Create a dummy image with colors
img = np.zeros((100, 100, 3), dtype=np.uint8)
# Red blob (Meat)
cv2.circle(img, (50, 50), 20, (0, 0, 255), -1) 
# Green blob (Veg)
cv2.circle(img, (20, 20), 10, (0, 255, 0), -1)
cv2.imwrite('test_image.png', img)

url = 'http://localhost:8000/predict'
files = {'file': open('test_image.png', 'rb')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
