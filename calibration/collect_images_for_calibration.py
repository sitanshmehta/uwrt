import cv2
import os

save_dir = "calibration_images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)  
i = 0

while i < 20:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Calibration Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        img_path = os.path.join(save_dir, f"image_{i}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        i += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
