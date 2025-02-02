import open3d as o3d
import numpy as np

def segment_bottle(point_cloud):
    # Segment plane (e.g., table) from point cloud
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000)
    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    # Perform clustering to find the bottle
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=50, print_progress=True))

    max_label = labels.max()
    bottle_cloud = None
    for i in range(max_label + 1):
        cluster = outlier_cloud.select_by_index(np.where(labels == i)[0])
        print(f"Cluster {i}: {len(cluster.points)} points")
        if len(cluster.points) > 5000:  # Assumption: bottle will have a significant number of points
            bottle_cloud = cluster
            break

    return bottle_cloud

# Load the combined point cloud
combined_point_cloud = o3d.io.read_point_cloud('path/to/combined_point_cloud.ply')

# Downsample the combined point cloud
combined_point_cloud = combined_point_cloud.voxel_down_sample(voxel_size=0.005)

# Save the downsampled combined point cloud
o3d.io.write_point_cloud('downsampled_combined_point_cloud.ply', combined_point_cloud)

# Segment the bottle from the combined point cloud
bottle_cloud = segment_bottle(combined_point_cloud)
if bottle_cloud:
    o3d.visualization.draw_geometries([bottle_cloud])
    o3d.io.write_point_cloud('bottle_point_cloud.ply', bottle_cloud)
else:
    print("Bottle not found in the point cloud.")
