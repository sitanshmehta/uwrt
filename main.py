import cv2
import torch
from ultralytics import YOLO
import numpy as np

keyboard_real_world = np.array([
    [0, 0],          # Top-left (origin)
    [26, 0],         # Top-right (26cm wide)
    [26, 9.3],       # Bottom-right
    [0, 9.3]         # Bottom-left
], dtype=np.float32)

def order_points(pts):
    # Sort points to [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]  # Bottom-right (largest sum)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right (smallest diff)
    rect[3] = pts[np.argmax(diff)]  # Bottom-left (largest diff)
    return rect

def main():
    model = YOLO("/Users/sitanshmehta/uwrt/20250308-training-70epoch/yolo11n.pt ")  
    
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
                if label == target:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # bounding boxes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    keyboard_roi = frame[y1:y2, x1:x2]
            
                    # Edge detection and contour extraction
                    gray = cv2.cvtColor(keyboard_roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                    if contours:
                        # Find the largest contour (keyboard)
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Approximate contour to 4 corners
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        if len(approx) == 4:
                            # Reshape and order the corners
                            corners_roi = approx.reshape(4, 2).astype(np.float32)
                            ordered_corners_roi = order_points(corners_roi)
                            
                            # Compute homography matrix
                            H, _ = cv2.findHomography(keyboard_real_world, ordered_corners_roi)
                            
                            # Optional: Refine corners using homography (if needed)
                            # projected_corners = cv2.perspectiveTransform(keyboard_real_world[None, ...], H)
                            
                            # Convert ROI corners back to original image coordinates
                            ordered_corners_image = ordered_corners_roi + [x1, y1]
                            
                            # Draw corners on the original frame
                            for (x, y) in ordered_corners_image.astype(int):
                                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                            
                            # Compute keyboard center
                            center = np.mean(ordered_corners_image, axis=0).astype(int)
                            cv2.circle(frame, tuple(center), 5, (255, 0, 0), -1)

        cv2.imshow("Keyboard Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
