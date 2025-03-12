import cv2
import numpy as np
import glob
import os

pattern_size = (2, 4)  
square_size = 0.050

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

objpoints = []  
imgpoints = []

images = glob.glob(os.path.join("calibration_images", "*.jpg"))
# print("Found images:", images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        print(f"✅ Corners detected in {fname}")
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        print(f"❌ Failed to detect corners in {fname}")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

np.savez("webcam_calib.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

with open("calibration_results.txt", "w") as f:
    f.write("Camera Matrix:\n")
    f.write(str(camera_matrix) + "\n")
    f.write("Distortion Coefficients:\n")
    f.write(str(dist_coeffs) + "\n")

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
