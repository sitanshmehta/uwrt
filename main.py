import cv2
import numpy as np
import imutils

def detect_keyboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    keyboard_contour = None
    max_area = 0
    
    # Iterate over contours to find the largest rectangular one (keyboard)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        
        if len(approx) >= 4 and area > max_area:  # Looking for a large quadrilateral shape
            keyboard_contour = approx
            max_area = area
    
    # Draw the detected keyboard contour
    if keyboard_contour is not None:
        cv2.drawContours(frame, [keyboard_contour], -1, (0, 255, 0), 3)
    
    return frame

def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = detect_keyboard(frame)
        
        # Display the result
        cv2.imshow("Keyboard Detection", processed_frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
