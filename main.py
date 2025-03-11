import cv2
import torch
from ultralytics import YOLO

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
        
        cv2.imshow("Keyboard Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
