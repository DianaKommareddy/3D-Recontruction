'''To convert depth and RGB images into a point cloud using Python, you can use libraries such as OpenCV and Open3D'''
'''import numpy as np
import cv2
import open3d as o3d

def create_point_cloud_from_rgb_depth(rgb_image_path, depth_image_path, intrinsics):
    # Load RGB and Depth images
    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # Ensure the depth image is in the same resolution as the RGB image
    depth_image = cv2.resize(depth_image, (rgb_image.shape[1], rgb_image.shape[0]))

    # Camera intrinsic parameters
    fx, fy, cx, cy = intrinsics

    # Create a point cloud array
    height, width = depth_image.shape
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            z = depth_image[v, u] / 1000.0  # Convert depth to meters
            if z == 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])
            colors.append(rgb_image[v, u] / 255.0)  # Normalize RGB values

    points = np.array(points)
    colors = np.array(colors)

    # Create Open3D Point Cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

# Example usage
rgb_image_path = 'path/to/your/rgb_image.png'
depth_image_path = 'path/to/your/depth_image.png'
intrinsics = [fx, fy, cx, cy]  # Replace with your camera's intrinsic parameters

point_cloud = create_point_cloud_from_rgb_depth(rgb_image_path, depth_image_path, intrinsics)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])

# Save the point cloud to a file
o3d.io.write_point_cloud("output_point_cloud.ply", point_cloud)'''

import cv2
import numpy as np
import open3d as o3d

# Define the intrinsic parameters of the camera
fx = 580.946  # Focal length in x
fy = 575.855  # Focal length in y
cx = 350.91   # Principal point x
cy = 228.44   # Principal point y

# Use the intrinsic parameters
intrinsics = [fx, fy, cx, cy]

# Load the RGB and depth images
rgb_image = cv2.imread('rgb/rgb_image.png')
depth_image = cv2.imread('depth/depth_image.png', cv2.IMREAD_UNCHANGED)

# Function to generate a colored point cloud from RGB and depth images
def generate_colored_point_cloud(rgb_image, depth_image, intrinsics, scale=1.0):
    fx, fy, cx, cy = intrinsics
    height, width = depth_image.shape
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            z = depth_image[v, u] / scale
            if z == 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append((x, y, z))
            colors.append(rgb_image[v, u] / 255.0)
    
    return np.array(points), np.array(colors)

# Generate the colored point cloud
scale_factor = 0.001  # Scale factor to convert depth values to meters (or adjust as needed)
points, colors = generate_colored_point_cloud(rgb_image, depth_image, intrinsics, scale=scale_factor)

# Create an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save the point cloud to a PLY file
output_ply_filename = 'colored_point_cloud.ply'
o3d.io.write_point_cloud(output_ply_filename, pcd)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="Colored Point Cloud", width=800, height=600)

# Save the point cloud visualization as an image
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(pcd)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("colored_point_cloud_output_image.png")
vis.destroy_window()

