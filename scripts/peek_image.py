import cv2, pathlib as p
im = cv2.imread(str(next((p.Path("data/processed/train")).glob("*.png"))))
print("shape:", im.shape)  # expect (224, 224, 3)
