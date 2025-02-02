import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Load KITTI dataset images
def load_images(path):
    image_files = sorted(glob.glob(path + "/*.png"))
    images = [cv2.imread(img, 0) for img in image_files]  # Load as grayscale
    return images

# Load calibration data (intrinsic parameters)
def load_calibration(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        P0 = np.array([float(x) for x in lines[0].strip().split()[1:]]).reshape(3, 4)
        K = P0[:, :3]  # 3x3 intrinsic matrix
    return K

# Find feature matches between two images
def feature_matching(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

# Compute camera motion
def compute_motion(kp1, kp2, matches, K):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def main():
    # Path to KITTI dataset
    right_image_path = "C:/Users/Diana mary/OneDrive/Desktop/odometry/dataset/sequences/14/image_0"
    left_image_path = "C:/Users/Diana mary/OneDrive/Desktop/odometry/dataset/sequences/14/image_1"
    calib_file = "C:/Users/Diana mary/OneDrive/Desktop/odometry/dataset/sequences/14/calib.txt"
    
    # Load images and calibration data
    left_images = load_images(left_image_path)
    right_images = load_images(right_image_path)
    K = load_calibration(calib_file)
    
    # Initial pose
    trajectory = np.zeros((3, 1))
    pose = np.eye(4)
    
    for i in range(len(left_images) - 1):
        img1 = left_images[i]
        img2 = left_images[i + 1]
        
        kp1, kp2, matches = feature_matching(img1, img2)
        if len(matches) > 8:
            R, t = compute_motion(kp1, kp2, matches, K)
            # Update pose
            Rt = np.eye(4)
            Rt[:3, :3] = R
            Rt[:3, 3] = t.squeeze()
            pose = pose @ np.linalg.inv(Rt)
            trajectory = np.hstack((trajectory, pose[:3, 3].reshape(3, 1)))
        
            # Draw keypoints and matches
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            plt.figure("Feature Matches")
            plt.imshow(img_matches)
            plt.title(f'Feature Matches Frame {i}')
            plt.show()
            plt.pause(0.01)
        
        # Print pose for debugging purposes
        x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
        print(f"Frame {i}: x = {x:.2f}, y = {y:.2f}, z = {z:.2f}")
    
    # Display final trajectory
    plt.figure("Estimated Trajectory")
    plt.plot(trajectory[0, :], trajectory[2, :], label="Estimated Trajectory")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
