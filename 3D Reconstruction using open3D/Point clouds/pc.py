import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load RGB and depth images
color_raw = o3d.io.read_image("RGB/rgbd_image_3.png")
depth_raw = o3d.io.read_image("depth/depth_image_3.png")

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
width = 640  # Adjust to your image width
height = 480  # Adjust to your image height

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
def save_point_cloud_image(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()

# Save the point cloud visualization
save_point_cloud_image(pcd, "point_cloud_output_image.png")
