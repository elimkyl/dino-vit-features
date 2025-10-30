import cv2
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from correspondences import find_correspondences, draw_correspondences
from visualize3d import print_3d_frame


num_pairs = 4
load_size = 112
layer = 9
facet = 'key' 
bin=True 
thresh=0.05
model_type='dino_vits8' 
stride=4 
H, W = 360, 640
how_far_from_webcam = 0.3 # meters

input_char = input("Press 'c' to start data collection or 's' to skip ...")
COLLECT_DATA = True if input_char == 'c' else False

if COLLECT_DATA:
    print("Starting data collection. Please 'q' to quit camera.")
    # Initialize camera
    os.makedirs("robot_demo_results", exist_ok=True)
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Camera Preview", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    if not cam.isOpened():
        raise IOError("Cannot open webcam")
    # Optional: wait a bit for the camera to adjust
    time.sleep(1)
    # Capture frame
    ret, img = cam.read()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Show the live camera feed
        cv2.imshow("Camera Preview", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            print("Exiting...")
            break
        # key == 32:  # Spacebar to capture
        elif key == ord('d'):  # 'd' to capture
            filename = "robot_demo_results/demo.jpg"
            size_down_img = cv2.resize(frame, (W, H))
            cv2.imwrite(filename, size_down_img)
            print(f"Image saved as {filename}. Please 'q' to quit camera.")
        elif key == ord('l'):
            filename = "robot_demo_results/live.jpg"
            size_down_img = cv2.resize(frame, (W, H))
            cv2.imwrite(filename, size_down_img)
            print(f"Image saved as {filename}. Please 'q' to quit camera.")

    # Release camera and close windows
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    time.sleep(1)

_ = input("Data Collection is Done! Press Enter to compute correspondences...")

# Assume demo and live images are captured at the same plane
rgb_bn = "robot_demo_results/demo.jpg"
rgb_live = "robot_demo_results/live.jpg"
depth = np.full((H, W), how_far_from_webcam, dtype=np.float32)  # meters
# Assume mac webcam intrinsics
intrinsics = {
    'fx': 0.8*W,
    'fy': 0.8*H,
    'cx': W / 2,
    'cy': H / 2
}

#Compute pixel correspondences between new observation and bottleneck observation.
with torch.no_grad():
    # This function from an external library takes image paths as input. Therefore, store the paths of the
    # observations and then pass those
    points1, points2, image1_pil, image2_pil = find_correspondences(rgb_bn, rgb_live, num_pairs, load_size, layer,
                                                                                       facet, bin, thresh, model_type, stride)
print(f"Correspondences computed successfully. Point 1: {points1}, Point 2: {points2}")

# Visualize the correspondencesfig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
colors = list(mcolors.TABLEAU_COLORS.values()) 
axes[0].imshow(image1_pil)
for (y, x), c in zip(points1, colors):
    axes[0].scatter(x, y, color=c, s=60, edgecolors='white', linewidth=1.5)
axes[0].set_title("Image 1")
axes[0].axis("off")

axes[1].imshow(image2_pil)
for (y, x), c in zip(points2, colors):
    axes[1].scatter(x, y, color=c, s=60, edgecolors='white', linewidth=1.5)
axes[1].set_title("Image 2")
axes[1].axis("off")
plt.savefig("robot_demo_results/correspondences.png")
plt.show()

def project_to_3d(points, depth, intrinsics):
    """
    Projects 2D pixel coordinates to 3D points using depth and camera intrinsics.

    Args:
        points (list of tuples): [(x, y), ...] pixel coordinates
        depth (np.ndarray): (H, W) or (H, W, 1) depth map in meters
        intrinsics (dict): camera intrinsics with keys 'fx', 'fy', 'cx', 'cy'

    Returns:
        point_with_depth: list of [X, Y, Z] 3D coordinates
    """
    point_with_depth = []

    for (u, v) in points:
        z = depth[v, u]  # depth at pixel (v, u)
        if z == 0:
            # Skip invalid depth
            continue
        # Reproject to 3D
        x = (u - intrinsics['cx']) * z / intrinsics['fx']
        y = (v - intrinsics['cy']) * z / intrinsics['fy']
        point_with_depth.append([x, y, z])

    return np.array(point_with_depth)

def find_transformation(X, Y):
    """
    Inputs: X, Y: lists of 3D points
    Outputs: R - 3x3 rotation matrix, t - 3-dim translation array.
    Find transformation given two sets of correspondences between 3D points.
    """
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t


#Given the pixel coordinates of the correspondences, and their depth values,
#project the points to 3D space.
points1 = project_to_3d(points1, depth, intrinsics)
points2 = project_to_3d(points2, depth, intrinsics)

#Find rigid translation and rotation that aligns the points by minimising error, using SVD.
R, t = find_transformation(points1, points2)

print("Transformation for the robot arm to match the demo pose:")
print("Estimated Rotation:\n", R)
print("Estimated Translation:\n", t)
print_3d_frame(R=R, t=t)
# Uncomment the line below to save the 3D frame visualization as an image.
# print_3d_frame(R=R, t=t, save_path="robot_demo_results/3d_frame.png")