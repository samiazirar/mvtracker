#!/usr/bin/env python3
"""
Point cloud processing utilities.

This module provides functions for working with 3D point clouds including:
- unproject_to_world_o3d: Convert depth maps to world-space point clouds
- clean_point_cloud_radius: Remove outliers using radius-based filtering
- reconstruct_mesh_from_pointcloud: Generate mesh from point cloud via Poisson
- _extract_points_inside_bbox: Extract points within an oriented bounding box
- _filter_points_closest_to_bbox: Keep N points closest to bbox center
- _exclude_points_inside_bbox: Remove points within a bounding box
- _filter_points_by_color_cluster: Filter points using DBSCAN color clustering
- _restrict_query_points_to_frames: Limit query points to specific frame range
- _align_bbox_with_point_cloud_com: Align bbox with nearby point cloud center of mass
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

def unproject_to_world_o3d(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, E_inv: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Creates a colored point cloud in world coordinates from a single view using Open3D.

    Args:
        depth: The depth map (H, W).
        rgb: The color image (H, W, 3).
        K: The 3x3 intrinsic camera matrix.
        E_inv: The 4x4 inverse extrinsic matrix (camera-to-world transformation).

    Returns:
        An Open3D PointCloud object in world coordinates.
    """
    o3d_depth = o3d.geometry.Image(depth)
    o3d_rgb = o3d.geometry.Image(rgb)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    
    H, W = depth.shape
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    
    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    
    # The point cloud is created in the camera's local coordinate system.
    # Transform it to the global world coordinate system.
    return pcd_cam.transform(E_inv)


def clean_point_cloud_radius(pcd: o3d.geometry.PointCloud, radius: float, min_points: int) -> o3d.geometry.PointCloud:
    """Radius-based outlier removal leveraging Open3D's built-in filter."""
    if not pcd.has_points():
        return pcd

    try:
        _, ind = pcd.remove_radius_outlier(nb_points=max(1, int(min_points)), radius=float(radius))
    except Exception as exc:  # pragma: no cover - Open3D failures are best-effort
        print(f"[WARN] Radius-based point cloud cleaning failed ({exc}); skipping filter.")
        return pcd

    if len(ind) == 0:
        print("[WARN] Radius-based point cloud cleaning removed all points; keeping original cloud.")
        return pcd

    return pcd.select_by_index(ind)


def reconstruct_mesh_from_pointcloud(pcd: o3d.geometry.PointCloud, depth: int) -> Optional[o3d.geometry.TriangleMesh]:
    """Reconstructs a mesh via Poisson surface reconstruction to sharpen geometry."""
    if not pcd.has_points():
        return None

    pcd_for_mesh = o3d.geometry.PointCloud(pcd)
    try:
        pcd_for_mesh.estimate_normals()
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"[WARN] Normal estimation failed ({exc}); skipping mesh reconstruction.")
        return None

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_for_mesh, depth=int(depth))
    except Exception as exc:  # pragma: no cover - Poisson may fail on sparse clouds
        print(f"[WARN] Mesh reconstruction failed ({exc}); skipping Poisson meshing.")
        return None

    densities_arr = np.asarray(densities)
    if densities_arr.size:
        density_thresh = np.quantile(densities_arr, 0.01)
        mask = densities_arr < density_thresh
        mesh.remove_vertices_by_mask(mask)

    if len(mesh.vertices) == 0:
        return None

    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _rotation_matrix_to_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion [x, y, z, w]."""
    # Validate the input early so downstream math can assume a proper rotation matrix.
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3.")
    trace = np.trace(R)
    if trace > 0.0:
        # Positive trace gives the most numerically stable branch; compute quaternion directly.
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        # Otherwise pick the dominant diagonal entry to compute the quaternion reliably.
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        # Degenerate rotation: fall back to identity quaternion.
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def _quaternion_xyzw_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion [x, y, z, w] to a rotation matrix."""
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[0] != 4:
        raise ValueError("Quaternion must have four components [x, y, z, w].")
    x, y, z, w = quat
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float32)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _extract_points_inside_bbox(
    points: np.ndarray,
    bbox: Dict[str, np.ndarray],
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract points that fall inside an oriented bounding box.
    
    Args:
        points: Nx3 array of 3D points in world coordinates
        bbox: Dictionary containing 'center', 'half_sizes', and 'basis' (or 'quat_xyzw')
        colors: Optional Nx3 array of RGB colors (0-1 range)
        
    Returns:
        Tuple of (inside_points, inside_colors) where inside_points is Mx3 array
        of points inside the bbox, and inside_colors is Mx3 array of corresponding colors
        (or None if colors was None)
    """
    if points is None or len(points) == 0 or bbox is None:
        return np.empty((0, 3), dtype=np.float32), None if colors is None else np.empty((0, 3), dtype=np.float32)
    
    points = np.asarray(points, dtype=np.float32)
    
    # Get bbox parameters
    center = np.asarray(bbox["center"], dtype=np.float32)
    half_sizes = np.asarray(bbox["half_sizes"], dtype=np.float32)
    
    # Get rotation basis
    basis = bbox.get("basis")
    if basis is None:
        quat = bbox.get("quat_xyzw")
        if quat is None:
            return np.empty((0, 3), dtype=np.float32), None if colors is None else np.empty((0, 3), dtype=np.float32)
        basis = _quaternion_xyzw_to_rotation_matrix(quat)
    basis = np.asarray(basis, dtype=np.float32)
    
    # Transform points to bbox local coordinates
    # points_local = (points - center) @ basis
    points_centered = points - center[None, :]
    points_local = points_centered @ basis
    
    # Check if points are inside the bbox (compare with half_sizes)
    # A point is inside if |local_coord[i]| <= half_sizes[i] for all i
    inside_mask = np.all(np.abs(points_local) <= half_sizes[None, :], axis=1)
    
    inside_points = points[inside_mask]
    inside_colors = None if colors is None else colors[inside_mask]
    
    return inside_points, inside_colors


def _filter_points_closest_to_bbox(
    points: np.ndarray,
    bbox: Dict[str, np.ndarray],
    max_points: int,
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Filter points to keep only the N closest to a bounding box center.
    
    Args:
        points: Nx3 array of 3D points
        bbox: Dictionary containing 'center' of the bbox
        max_points: Maximum number of points to keep
        colors: Optional Nx3 array of RGB colors (0-1 range)
        
    Returns:
        Tuple of (filtered_points, filtered_colors)
    """
    if points is None or len(points) == 0 or bbox is None or max_points is None or max_points <= 0:
        return points, colors
    
    if len(points) <= max_points:
        return points, colors
    
    # Calculate distances to bbox center
    center = np.asarray(bbox["center"], dtype=np.float32)
    distances = np.linalg.norm(points - center[None, :], axis=1)
    
    # Get indices of N closest points
    closest_indices = np.argsort(distances)[:max_points]
    
    filtered_points = points[closest_indices]
    filtered_colors = None if colors is None else colors[closest_indices]
    
    return filtered_points, filtered_colors


def _exclude_points_inside_bbox(
    points: np.ndarray,
    bbox: Dict[str, np.ndarray],
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Exclude points that fall inside a bounding box (keep points outside).
    
    Args:
        points: Nx3 array of 3D points in world coordinates
        bbox: Dictionary containing 'center', 'half_sizes', and 'basis' (or 'quat_xyzw')
        colors: Optional Nx3 array of RGB colors (0-1 range)
        
    Returns:
        Tuple of (outside_points, outside_colors) where outside_points is Mx3 array
        of points outside the bbox, and outside_colors is Mx3 array of corresponding colors
        (or None if colors was None)
    """
    if points is None or len(points) == 0 or bbox is None:
        return points, colors
    
    points = np.asarray(points, dtype=np.float32)
    
    # Get bbox parameters
    center = np.asarray(bbox["center"], dtype=np.float32)
    half_sizes = np.asarray(bbox["half_sizes"], dtype=np.float32)
    
    # Get rotation basis
    basis = bbox.get("basis")
    if basis is None:
        quat = bbox.get("quat_xyzw")
        if quat is None:
            return points, colors
        basis = _quaternion_xyzw_to_rotation_matrix(quat)
    basis = np.asarray(basis, dtype=np.float32)
    
    # Transform points to bbox local coordinates
    points_centered = points - center[None, :]
    points_local = points_centered @ basis
    
    # Check if points are OUTSIDE the bbox (inverse of inside check)
    # A point is outside if ANY |local_coord[i]| > half_sizes[i]
    outside_mask = np.any(np.abs(points_local) > half_sizes[None, :], axis=1)
    
    outside_points = points[outside_mask]
    outside_colors = None if colors is None else colors[outside_mask]
    
    return outside_points, outside_colors


def _filter_points_by_color_cluster(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    eps: float = 0.15,
    min_samples: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Filter points using DBSCAN clustering on colors. Keeps only the largest cluster.
    Useful for filtering out non-gripper colored points (gripper is typically black/dark).
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (0-1 range). Required for clustering.
        eps: Maximum distance between samples for DBSCAN (color space)
        min_samples: Minimum samples in neighborhood for DBSCAN
        
    Returns:
        Tuple of (filtered_points, filtered_colors) containing only largest cluster
    """
    if points is None or len(points) == 0 or colors is None or len(colors) == 0:
        return points, colors
    
    if len(points) != len(colors):
        print("[WARN] Points and colors size mismatch in color clustering; skipping filter")
        return points, colors
    
    # Convert colors to numpy array if needed
    colors_arr = np.asarray(colors, dtype=np.float32)
    
    # Perform DBSCAN clustering on color values
    # eps is in color space (0-1 range), so 0.15 means colors within ~38/255 distance
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(colors_arr)
    labels = clustering.labels_
    
    # Find the largest cluster (excluding noise which has label -1)
    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1)
    
    if len(valid_labels) == 0:
        print("[WARN] No valid clusters found in color clustering; keeping all points")
        return points, colors
    
    # Count points in each cluster
    cluster_sizes = {label: np.sum(labels == label) for label in valid_labels}
    largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
    
    # Filter to keep only largest cluster
    cluster_mask = labels == largest_cluster_label
    filtered_points = points[cluster_mask]
    filtered_colors = colors_arr[cluster_mask]
    
    print(f"[INFO] Color clustering: kept {len(filtered_points)}/{len(points)} points from largest cluster (label {largest_cluster_label})")
    
    return filtered_points, filtered_colors


def _restrict_query_points_to_frames(
    query_points: Optional[List[Optional[np.ndarray]]],
    query_colors: Optional[List[Optional[np.ndarray]]],
    max_frames: Optional[int],
) -> Tuple[Optional[List[Optional[np.ndarray]]], Optional[List[Optional[np.ndarray]]]]:
    """
    Limit query point exports to the first `max_frames` frames while leaving the processing pipeline untouched.

    Args:
        query_points: Per-frame list of query point arrays (or None) assembled during processing.
        query_colors: Optional per-frame list of color arrays aligned with `query_points`.
        max_frames: If positive, only frames [0, max_frames) retain their query points; later frames are cleared.

    Returns:
        Tuple containing filtered versions of `query_points` and `query_colors`.
    """
    if query_points is None or max_frames is None:
        return query_points, query_colors

    # Clamp non-positive requests to zero frames.
    if max_frames <= 0:
        max_frames = 0

    filtered_points = list(query_points)
    filtered_colors = list(query_colors) if query_colors is not None else None

    for idx in range(max_frames, len(filtered_points)):
        filtered_points[idx] = None
        if filtered_colors is not None and idx < len(filtered_colors):
            filtered_colors[idx] = None

    return filtered_points, filtered_colors


def _align_bbox_with_point_cloud_com(
    bbox: Dict[str, np.ndarray],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    search_radius_scale: float = 2.0,
    min_points_required: int = 10,
) -> Optional[Dict[str, np.ndarray]]:
    """Align a bbox with the center of mass of nearby point-cloud samples (x/y translation, planar rotation)."""
    if bbox is None or points is None or len(points) == 0:
        return bbox

    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < min_points_required:
        return bbox

    if colors is not None and colors.shape[0] not in (0, points.shape[0]):
        raise NotImplementedError("this needs some serious fixes.")

    center = np.asarray(bbox["center"], dtype=np.float32)
    half_sizes = np.asarray(bbox["half_sizes"], dtype=np.float32)
    basis = np.asarray(bbox.get("basis", np.eye(3, dtype=np.float32)), dtype=np.float32)

    bbox_diagonal = np.linalg.norm(half_sizes * 2.0)
    search_radius = bbox_diagonal * search_radius_scale

    distances = np.linalg.norm(points - center[None, :], axis=1)
    nearby_mask = distances <= search_radius
    nearby_points = points[nearby_mask]
    if nearby_points.shape[0] < min_points_required:
        return bbox

    com = np.mean(nearby_points, axis=0).astype(np.float32)
    aligned_center = center.copy()
    # Only slide the box in the plane; trust the original Z so we do not fight gravity offsets.
    aligned_center[0] = com[0]
    aligned_center[1] = com[1]

    points_xy = nearby_points[:, :2] - com[:2]
    approach_axis = basis[:, 2].astype(np.float32)
    approach_norm = np.linalg.norm(approach_axis)
    if approach_norm < 1e-12:
        return {
            "center": aligned_center,
            "half_sizes": half_sizes,
            "quat_xyzw": bbox.get("quat_xyzw"),
            "basis": basis,
        }
    approach_axis /= approach_norm
    width_axis = basis[:, 0].astype(np.float32)
    height_axis = basis[:, 1].astype(np.float32)

    if points_xy.shape[0] >= 3:
        cov = np.cov(points_xy.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            dominant_xy = eigenvectors[:, 0]
            width_xy = width_axis[:2]
            width_xy_norm = np.linalg.norm(width_xy)
            dominant_norm = np.linalg.norm(dominant_xy)

            if width_xy_norm > 1e-9 and dominant_norm > 1e-9:
                current_angle = float(np.arctan2(width_xy[1], width_xy[0]))
                target_angle = float(np.arctan2(dominant_xy[1], dominant_xy[0]))
                delta = target_angle - current_angle
                delta = (delta + np.pi) % (2.0 * np.pi) - np.pi

                if abs(delta) > 1e-6:
                    axis = approach_axis
                    cos_d = np.cos(delta)
                    sin_d = np.sin(delta)

                    def _rotate(vec: np.ndarray) -> np.ndarray:
                        # Rotate vec around axis by angle delta using Rodrigues formula.
                        cross_part = np.cross(axis, vec)
                        dot_part = np.dot(axis, vec)
                        return (
                            vec * cos_d
                            + cross_part * sin_d
                            + axis * dot_part * (1.0 - cos_d)
                        ).astype(np.float32)

                    width_axis = _rotate(width_axis)
                    height_axis = _rotate(height_axis)
        except np.linalg.LinAlgError:
            pass

    width_axis_norm = np.linalg.norm(width_axis)
    if width_axis_norm < 1e-12:
        width_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if np.isfinite(width_axis_norm) and np.abs(np.dot(width_axis, approach_axis)) > 0.9:
            width_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    width_axis /= np.linalg.norm(width_axis) + 1e-12

    # Ensure height axis stays orthogonal to both width and approach directions.
    height_axis = height_axis - np.dot(height_axis, approach_axis) * approach_axis
    height_axis_norm = np.linalg.norm(height_axis)
    if height_axis_norm < 1e-12:
        height_axis = np.cross(approach_axis, width_axis)
    height_axis /= np.linalg.norm(height_axis) + 1e-12

    approach_axis_new = np.cross(width_axis, height_axis)
    approach_axis_new /= np.linalg.norm(approach_axis_new) + 1e-12

    if np.dot(approach_axis_new, approach_axis) < 0.0:
        approach_axis_new = -approach_axis_new
        height_axis = -height_axis

    aligned_basis = np.column_stack((width_axis, height_axis, approach_axis_new)).astype(np.float32)
    aligned_quat = _rotation_matrix_to_xyzw(aligned_basis)

    return {
        "center": aligned_center,
        "half_sizes": half_sizes,
        "quat_xyzw": aligned_quat,
        "basis": aligned_basis,
    }
