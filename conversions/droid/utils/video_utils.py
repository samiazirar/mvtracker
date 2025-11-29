import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from .transforms import transform_points, invert_transform

class VideoRecorder:
    def __init__(self, output_dir, cam_serial, suffix, width, height, fps=15):
        self.filepath = os.path.join(output_dir, f"{cam_serial}_{suffix}.mp4")
        os.makedirs(output_dir, exist_ok=True)
        self.writer = None
        self.needs_conversion = False
        self.filepath_temp = None

        # Try XVID first (widely supported), we'll convert to H.264 later
        for codec in ['XVID', 'mp4v', 'MJPG']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                if codec == 'MJPG':
                    # MJPG needs .avi extension
                    self.filepath_temp = self.filepath.replace('.mp4', '_temp.avi')
                    writer = cv2.VideoWriter(self.filepath_temp, fourcc, fps, (width, height))
                    self.needs_conversion = True
                else:
                    writer = cv2.VideoWriter(self.filepath, fourcc, fps, (width, height))
                    
                if writer.isOpened():
                    self.writer = writer
                    break
            except Exception:
                continue
        
        if self.writer is None:
            print(f"[WARN] Failed to initialize VideoWriter for {self.filepath}")
        
    def write_frame(self, image):
        if self.writer:
            self.writer.write(image)
        
    def close(self):
        if self.writer:
            self.writer.release()
        
        # Convert to H.264 for better compatibility
        source = self.filepath_temp if self.needs_conversion else self.filepath
        if source and os.path.exists(source) and os.path.getsize(source) > 0:
            import subprocess
            temp_out = self.filepath + ".tmp.mp4"
            try:
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', source,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    temp_out
                ], capture_output=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(temp_out):
                    # Replace original with converted
                    if self.needs_conversion and os.path.exists(self.filepath_temp):
                        os.remove(self.filepath_temp)
                    if os.path.exists(self.filepath):
                        os.remove(self.filepath)
                    os.rename(temp_out, self.filepath)
            except Exception as e:
                # Keep original file if conversion fails
                if os.path.exists(temp_out):
                    os.remove(temp_out)


def project_points_to_image_fast(points_3d, K, T_world_cam, width, height, colors=None):
    """
    Fast vectorized projection of 3D world points onto the camera image plane.
    
    Args:
        points_3d: Nx3 numpy array of points in World Frame.
        K: 3x3 Intrinsic matrix.
        T_world_cam: 4x4 Transformation matrix (Camera Pose in World).
        width: Image width.
        height: Image height.
        colors: Optional Nx3 numpy array of colors corresponding to points.
        
    Returns:
        Nx2 numpy array of (u, v) coordinates as int32.
        (Optional) Nx3 numpy array of colors if colors was provided.
    """
    if points_3d is None or len(points_3d) == 0:
        return (np.array([], dtype=np.int32).reshape(-1, 2), np.array([])) if colors is not None else np.array([], dtype=np.int32).reshape(-1, 2)

    # Transform World -> Camera (vectorized)
    T_cam_world = invert_transform(T_world_cam)
    
    # Fast transform: avoid function call overhead
    points_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_cam = (T_cam_world @ points_homo.T).T[:, :3]
    
    # Filter points behind camera (z <= 0.1)
    z = points_cam[:, 2]
    mask = z > 0.1
    points_cam = points_cam[mask]
    z = z[mask]
    
    if colors is not None:
        colors = colors[mask]
    
    if len(points_cam) == 0:
        return (np.array([], dtype=np.int32).reshape(-1, 2), np.array([])) if colors is not None else np.array([], dtype=np.int32).reshape(-1, 2)

    # Project: vectorized (x, y, z) -> (u, v)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = (points_cam[:, 0] * fx / z) + cx
    v = (points_cam[:, 1] * fy / z) + cy
    
    # Filter points outside image bounds (vectorized)
    valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    u = u[valid_mask].astype(np.int32)
    v = v[valid_mask].astype(np.int32)
    uv = np.column_stack((u, v))
                 
    if colors is not None:
        return uv, colors[valid_mask]
    return uv


def draw_points_on_image_fast(image, points_2d, color=(0, 255, 0), radius=1, colors=None):
    """
    Fast vectorized point drawing on an image using numpy indexing.
    
    For radius=1, uses direct pixel assignment which is much faster than cv2.circle.
    For radius>1, falls back to cv2.circle but batched.
    """
    if len(points_2d) == 0:
        return image
    
    img_out = image.copy()
    h, w = img_out.shape[:2]
    
    # Ensure integer coordinates
    pts = points_2d.astype(np.int32)
    
    if radius == 1:
        # Fast path: direct pixel assignment (vectorized)
        # Clip to valid range
        valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        pts = pts[valid]
        
        if colors is not None:
            cols = colors[valid].astype(np.uint8)
            # BGR order for OpenCV
            img_out[pts[:, 1], pts[:, 0]] = cols
        else:
            img_out[pts[:, 1], pts[:, 0]] = color
    else:
        # Fallback: use cv2.circle for larger radius
        if colors is not None:
            colors = colors.astype(np.uint8)
            for pt, col in zip(pts, colors):
                cv2.circle(img_out, (int(pt[0]), int(pt[1])), radius, 
                          (int(col[0]), int(col[1]), int(col[2])), -1)
        else:
            for pt in pts:
                cv2.circle(img_out, (int(pt[0]), int(pt[1])), radius, color, -1)
            
    return img_out


# Keep original functions for backward compatibility
def project_points_to_image(points_3d, K, T_world_cam, width, height, colors=None):
    """Original projection function - kept for compatibility."""
    return project_points_to_image_fast(points_3d, K, T_world_cam, width, height, colors)


def draw_points_on_image(image, points_2d, color=(0, 255, 0), radius=1, colors=None):
    """Original drawing function - kept for compatibility."""
    return draw_points_on_image_fast(image, points_2d, color, radius, colors)


def process_and_write_frame(args):
    """
    Process a single camera's frame and return the result.
    Used for parallel processing.
    """
    serial, img, pts_world, cols_world, K, w, h, T_cam, recorder = args
    
    if T_cam is not None and len(pts_world) > 0:
        uv, cols = project_points_to_image_fast(pts_world, K, T_cam, w, h, colors=cols_world)
        img_out = draw_points_on_image_fast(img, uv, colors=cols)
        recorder.write_frame(img_out)
    else:
        recorder.write_frame(img)
    
    return serial
