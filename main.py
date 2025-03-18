import cv2
import torch
from ultralytics import YOLO
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)  # Create logs directory if needed

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"log_{timestamp}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_file),  
        logging.StreamHandler()         
    ]
)

#idea: keyboard should be at z = 0, and is pointing in the positive z direction. 
#camera needs to be pointing in the -z direction so that it is looking at the keyboard

fx = 1.06132388e+03
cx = 6.27894027e+02
fy = 1.06182789e+03
cy = 3.02051426e+02

k1 = 0.22129463
k2 = -0.28233807
p1 = -0.019219
p2 =  -0.01305288
k3 = -1.29031665

# Camera calibration data (REPLACE WITH YOUR VALUES)
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Keyboard 3D corners (cm, z=0)
keyboard_real_world = np.array([
    [0, 0, 0],       # Top-left
    [26, 0, 0],      # Top-right
    [26, 9.3, 0],    # Bottom-right
    [0, 9.3, 0]      # Bottom-left
], dtype=np.float32)

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def draw_axis(img, corner, imgpts):
    corner = tuple(corner.astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    return img

def main():
    model = YOLO("/Users/sitanshmehta/uwrt/20250308-training-70epoch/yolo11n.pt")  
    target = "keyboard" 
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                if label == target and box.conf[0] > 0.5:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    keyboard_roi = frame[y1:y2, x1:x2]
                    gray = cv2.cvtColor(keyboard_roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        if len(approx) == 4:
                            corners_roi = approx.reshape(4, 2).astype(np.float32)
                            ordered_corners_roi = order_points(corners_roi)
                            ordered_corners_image = ordered_corners_roi + [x1, y1]
                            
                            # Compute 3D pose
                            success, rvec, tvec = cv2.solvePnP(
                                keyboard_real_world,
                                ordered_corners_image.astype(np.float32),
                                camera_matrix,
                                distortion_coeffs
                            )
                            
                            if success:
                                # Visualize 3D axes
                                axis_points = np.float32([[10,0,0], [0,10,0], [0,0,10]]).reshape(-1,3)
                                imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, distortion_coeffs)
                                frame = draw_axis(frame, ordered_corners_image[0], imgpts)
                                
                                # Compute desired camera movement
                                desired_rotation = -rvec
                                desired_translation = -tvec
                                print(f"Move camera by (x,y,z): {desired_translation}")
                                print(f"Rotate camera by (rx,ry,rz): {desired_rotation}")

        cv2.imshow("Keyboard Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()