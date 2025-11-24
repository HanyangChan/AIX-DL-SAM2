@echo off
echo Starting Training with 10% of data...
"C:\Users\a7334\AppData\Local\Programs\Python\Python313\python.exe" backend/train_classifier.py --data_dir backend/dataset/food_classification --epochs 10 --sample_ratio 0.1
pause
