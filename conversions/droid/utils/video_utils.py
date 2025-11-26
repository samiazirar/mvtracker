import cv2
import numpy as np
import os
from .transforms import transform_points, invert_transform

class VideoRecorder:
    def __init__(self, output_dir, cam_serial, suffix, width, height, fps=15):
        self.filepath = os.path.join(output_dir, f"{cam_serial}_{suffix}.mp4")
        os.makedirs(output_dir, exist_ok=True)
        # Try multiple codecs for better compatibility
        for codec in ['mp4v', 'XVID', 'MJPG', 'avc1']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(self.filepath, fourcc, fps, (width, height))
            if self.writer.isOpened():
                print(f"[VIDEO] Using codec '{codec}' for {cam_serial}_{suffix}")
                break
        else:
            raise RuntimeError(f"Failed to initialize VideoWriter for {self.filepath}. No compatible codec found.")
        
    def write_frame(self, image):
        self.writer.write(image)
        
    def close(self):
        self.writer.release()

def project_points_to_image(points_3d, K, T_world_cam, width, height):
    """
    Project 3D world points onto the camera image plane.
    
    Args:
        points_3d: Nx3 numpy array of points in World Frame.
        K: 3x3 Intrinsic matrix.
        T_world_cam: 4x4 Transformation matrix (Camera Pose in World).
        width: Image width.
        height: Image height.
        
    Returns:
        Nx2 numpy array of (u, v) coordinates.
    """
    if points_3d is None or len(points_3d) == 0:
        return np.array([])

    # Transform World -> Camera
    T_cam_world = invert_transform(T_world_cam)
    points_cam = transform_points(points_3d, T_cam_world)
    
    # Filter points behind camera (z <= 0.1)
    mask = points_cam[:, 2] > 0.1
    points_cam = points_cam[mask]
    
    if len(points_cam) == 0:
        return np.array([])

    # Project: (x, y, z) -> (u, v)
    # u = fx * x / z + cx
    # v = fy * y / z + cy
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z = points_cam[:, 2]
    u = (points_cam[:, 0] * fx / z) + cx
    v = (points_cam[:, 1] * fy / z) + cy
    
    # Stack and filter points outside image bounds
    uv = np.column_stack((u, v))
    
    valid_mask = (uv[:, 0] >= 0) & (uv[:, 0] < width) & \
                 (uv[:, 1] >= 0) & (uv[:, 1] < height)
                 
    return uv[valid_mask]

def draw_points_on_image(image, points_2d, color=(0, 255, 0), radius=1):
    """
    Draw points on an image.
    """
    if len(points_2d) == 0:
        return image
        
    img_copy = image.copy()
    # Draw all points at once? No, cv2.circle is one by one.
    # For speed, we could use direct pixel access or just loop.
    # Given the number of points might be large, let's try to be efficient.
    
    for pt in points_2d:
        cv2.circle(img_copy, (int(pt[0]), int(pt[1])), radius, color, -1)
    return img_copy
