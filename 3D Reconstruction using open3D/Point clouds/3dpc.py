import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load RGB and depth images
color_image_path = "RGB/rgbd_image_3.png"
depth_image_path = "depth/depth_image_3.png"

# Check if the paths are correct
try:
    color_raw = o3d.io.read_image(color_image_path)
    depth_raw = o3d.io.read_image(depth_image_path)
except Exception as e:
    print(f"Error reading images: {e}")
    exit()

# Check if images are loaded properly
if color_raw.is_empty() or depth_raw.is_empty():
    print("Failed to load images. Please check the paths and image files.")
    exit()

# Check image sizes
color_image_size = np.asarray(color_raw).shape
depth_image_size = np.asarray(depth_raw).shape

print(f"RGB image size: {color_image_size}")
print(f"Depth image size: {depth_image_size}")

# Ensure both images have the same dimensions
if color_image_size[:2] != depth_image_size[:2]:
    print("RGB and depth image sizes do not match. Please provide images with the same dimensions.")
    exit()

# Create an RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw
)
print(rgbd_image)

# Display the grayscale and depth images
plt.subplot(1, 2, 1)
plt.title("Grayscale image")
plt.imshow(np.asarray(rgbd_image.color))
plt.subplot(1, 2, 2)
plt.title("Depth image")
plt.imshow(np.asarray(rgbd_image.depth))
plt.show()

# Define the intrinsic parameters of the camera
fx = 580.946
fy = 575.855
cx = 350.91
cy = 228.44
width = color_image_size[1]  # Adjust to your image width
height = color_image_size[0]  # Adjust to your image height

intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Create a point cloud from the RGBD image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# Transform the point cloud to match the Open3D coordinate system
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud", width=800, height=600)

# Save the point cloud to a PLY file
output_ply_filename = 'depth_image_point_cloud_3.ply'
o3d.io.write_point_cloud(output_ply_filename, pcd)

# Save the point cloud visualization as an image
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(pcd)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("point_cloud_output_image.png")
vis.destroy_window()
