import requests
import cv2
import numpy as np

# Create a dummy image with colors
# Create a dummy image with colors
img = np.zeros((100, 100, 3), dtype=np.uint8)
# Red blob (Meat)
cv2.circle(img, (50, 50), 20, (0, 0, 255), -1) 
# Green blob (Veg)
cv2.circle(img, (20, 20), 10, (0, 255, 0), -1)

# Use Korean filename to test unicode handling
filename = '테스트_이미지.png'
# cv2.imwrite doesn't support unicode paths on windows well either, 
# so we write to ascii then rename or use imencode
success, buf = cv2.imencode(".png", img)
with open(filename, "wb") as f:
    f.write(buf)
import requests
import cv2
import numpy as np

# Create a dummy image with colors
# Create a dummy image with colors
img = np.zeros((100, 100, 3), dtype=np.uint8)
# Red blob (Meat)
cv2.circle(img, (50, 50), 20, (0, 0, 255), -1) 
# Green blob (Veg)
cv2.circle(img, (20, 20), 10, (0, 255, 0), -1)

# Use the multi-object image
filename = 'multi_object_test.png'

url = 'http://localhost:8000/predict'
files = {'file': (filename, open(filename, 'rb'), 'image/png')}

try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    try:
        data = response.json()
        print("Response Data:", data)
        print("Detected Items:")
        for item in data['food_items']:
            print(f"- {item['label']} ({item['calories']} kcal)")
    except:
        print("Response Text:", response.text)
except Exception as e:
    print("Error:", e)
