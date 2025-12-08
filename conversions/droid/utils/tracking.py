"""Utilities for gripper contact tracking and minimal track visualization."""

import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import rerun as rr

# Default mesh used for sampling contact points
DEFAULT_INNER_FINGER_MESH = "/workspace/third_party/robotiq_arg85_description/meshes/inner_finger_fine.STL"


def extract_contact_surface(mesh):
    """Extract the inner rubber contact pad (faces facing -Y)."""
    vertices = mesh.vertices
    contact_face_indices = []
    for i, face in enumerate(mesh.faces):
        face_verts = vertices[face]
        y_min = face_verts[:, 1].min()
        # The innermost flat surface at Y = -0.011
        if y_min < -0.0095:
            contact_face_indices.append(i)

    if len(contact_face_indices) == 0:
        print("[WARN] Could not extract contact surface for tracking")
        return None

    contact_submesh = mesh.submesh([contact_face_indices], only_watertight=False)[0]
    print(f"[INFO] Extracted contact pad: {len(contact_submesh.vertices)} verts, {len(contact_submesh.faces)} faces")
    return contact_submesh


def sample_contact_points(mesh, num_points):
    """Sample points uniformly on the contact surface."""
    if num_points <= len(mesh.vertices):
        # Use a deterministic subset when requesting fewer points
        points = mesh.vertices[:num_points].copy()
    else:
        # Allow oversampling beyond vertex count
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
    print(f"[INFO] Sampled {len(points)} contact points for tracking")
    return points


def compute_finger_transforms(T_base_ee, gripper_pos):
    """Compute transforms for left/right inner fingers."""
    val = gripper_pos[0] if isinstance(gripper_pos, (list, np.ndarray)) else gripper_pos
    theta = val * 0.8

    # Left finger chain: base_ee -> left_inner_knuckle -> left_inner_finger
    T_lik = np.eye(4)
    T_lik[:3, 3] = [0.0127, 0, 0.0693]
    T_lik[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()

    T_lif = np.eye(4)
    T_lif[:3, 3] = [0.03458531, 0, 0.04549702]
    T_lif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
    T_world_left_finger = T_base_ee @ T_lik @ T_lif

    # Right finger chain: base_ee -> right_inner_knuckle -> right_inner_finger
    T_rik = np.eye(4)
    T_rik[:3, 3] = [-0.0127, 0, 0.0693]
    T_rik[:3, :3] = R.from_euler('z', np.pi).as_matrix() @ R.from_rotvec([0, -theta, 0]).as_matrix()

    T_rif = np.eye(4)
    T_rif[:3, 3] = [0.03410605, 0, 0.04585739]
    T_rif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
    T_world_right_finger = T_base_ee @ T_rik @ T_rif

    return T_world_left_finger, T_world_right_finger


def compute_contact_frame(pts_left_world, pts_right_world, T_base_ee):
    """Compute a simplified pose (coordinate frame) for the contact points.

    The contact frame represents the center of mass of all contact points
    with an orientation derived from the gripper's end-effector orientation.

    Args:
        pts_left_world: [N, 3] contact points on left finger in world frame
        pts_right_world: [N, 3] contact points on right finger in world frame
        T_base_ee: [4, 4] end-effector pose (used for orientation)

    Returns:
        contact_centroid: [3] center of mass of all contact points
        contact_frame: [4, 4] homogeneous transform with centroid as origin
                       and orientation from the end-effector
    """
    # Combine all contact points
    all_pts = np.vstack([pts_left_world, pts_right_world])

    # Compute centroid (center of mass)
    contact_centroid = np.mean(all_pts, axis=0)

    # Build contact frame: position = centroid, orientation = from end-effector
    # This gives a stable orientation that follows the gripper
    contact_frame = np.eye(4)
    contact_frame[:3, :3] = T_base_ee[:3, :3]  # Use EE orientation
    contact_frame[:3, 3] = contact_centroid     # Position at centroid

    return contact_centroid, contact_frame


def compute_contact_frame_per_finger(pts_world, T_finger):
    """Compute a simplified pose for a single finger's contact points.

    Args:
        pts_world: [N, 3] contact points on finger in world frame
        T_finger: [4, 4] finger transform (used for orientation)

    Returns:
        centroid: [3] center of mass of contact points
        frame: [4, 4] homogeneous transform with centroid as origin
    """
    centroid = np.mean(pts_world, axis=0)

    frame = np.eye(4)
    frame[:3, :3] = T_finger[:3, :3]
    frame[:3, 3] = centroid

    return centroid, frame


def compute_normalized_flow(contact_centroids, contact_frames, step_size_mm=1.0,
                           tracks_3d=None, left_contact_frames=None, right_contact_frames=None):
    """Normalize contact flow to fixed distance steps (e.g., 1mm).

    Resamples the trajectory based on 3D distance traveled rather than time.
    - If robot moves 10mm in one frame, interpolates to 10 steps
    - If robot moves 1mm over 10 frames, combines into 1 step

    Args:
        contact_centroids: [T, 3] center of mass positions over time
        contact_frames: [T, 4, 4] contact frames over time
        step_size_mm: Target step size in millimeters (default: 1.0)
        tracks_3d: [T, N, 3] optional contact track points to also normalize
        left_contact_frames: [T, 4, 4] optional left finger frames to normalize
        right_contact_frames: [T, 4, 4] optional right finger frames to normalize

    Returns:
        dict with:
            normalized_centroids: [M, 3] resampled centroids at fixed distance steps
            normalized_frames: [M, 4, 4] resampled frames at fixed distance steps
            normalized_tracks_3d: [M, N, 3] resampled contact points (if tracks_3d provided)
            normalized_left_frames: [M, 4, 4] resampled left finger frames (if provided)
            normalized_right_frames: [M, 4, 4] resampled right finger frames (if provided)
            cumulative_distance_mm: [T] cumulative distance at each original frame (in mm)
            frame_to_normalized_idx: [T] mapping from original frame to normalized index
            num_normalized_steps: int number of normalized steps
    """
    step_size_m = step_size_mm / 1000.0  # Convert to meters

    T = len(contact_centroids)
    result = {}

    if T < 2:
        result['normalized_centroids'] = contact_centroids.copy()
        result['normalized_frames'] = contact_frames.copy()
        result['cumulative_distance_mm'] = np.zeros(T)
        result['frame_to_normalized_idx'] = np.zeros(T, dtype=np.int32)
        result['num_normalized_steps'] = T
        if tracks_3d is not None:
            result['normalized_tracks_3d'] = tracks_3d.copy()
        if left_contact_frames is not None:
            result['normalized_left_frames'] = left_contact_frames.copy()
        if right_contact_frames is not None:
            result['normalized_right_frames'] = right_contact_frames.copy()
        return result

    # Compute distances between consecutive frames
    diffs = np.diff(contact_centroids, axis=0)  # [T-1, 3]
    distances = np.linalg.norm(diffs, axis=1)   # [T-1]

    # Cumulative distance (in meters)
    cumulative_distance = np.zeros(T)
    cumulative_distance[1:] = np.cumsum(distances)

    # Total distance traveled
    total_distance = cumulative_distance[-1]

    if total_distance < step_size_m:
        # Very short trajectory - return start and end only
        result['normalized_centroids'] = np.array([contact_centroids[0], contact_centroids[-1]])
        result['normalized_frames'] = np.array([contact_frames[0], contact_frames[-1]])
        result['cumulative_distance_mm'] = cumulative_distance * 1000
        frame_to_normalized_idx = np.zeros(T, dtype=np.int32)
        frame_to_normalized_idx[T//2:] = 1
        result['frame_to_normalized_idx'] = frame_to_normalized_idx
        result['num_normalized_steps'] = 2
        if tracks_3d is not None:
            result['normalized_tracks_3d'] = np.array([tracks_3d[0], tracks_3d[-1]])
        if left_contact_frames is not None:
            result['normalized_left_frames'] = np.array([left_contact_frames[0], left_contact_frames[-1]])
        if right_contact_frames is not None:
            result['normalized_right_frames'] = np.array([right_contact_frames[0], right_contact_frames[-1]])
        return result

    # Number of normalized steps
    num_steps = int(np.ceil(total_distance / step_size_m)) + 1

    # Target distances for each normalized step
    target_distances = np.linspace(0, total_distance, num_steps)

    # Interpolate centroids at target distances
    normalized_centroids = np.zeros((num_steps, 3), dtype=np.float32)
    normalized_frames = np.zeros((num_steps, 4, 4), dtype=np.float32)

    # Optional: normalize tracks_3d
    if tracks_3d is not None:
        num_points = tracks_3d.shape[1]
        normalized_tracks_3d = np.zeros((num_steps, num_points, 3), dtype=np.float32)
    else:
        normalized_tracks_3d = None

    # Optional: normalize per-finger frames
    if left_contact_frames is not None:
        normalized_left_frames = np.zeros((num_steps, 4, 4), dtype=np.float32)
    else:
        normalized_left_frames = None

    if right_contact_frames is not None:
        normalized_right_frames = np.zeros((num_steps, 4, 4), dtype=np.float32)
    else:
        normalized_right_frames = None

    for i, target_dist in enumerate(target_distances):
        # Find the two original frames that bracket this distance
        idx = np.searchsorted(cumulative_distance, target_dist)

        if idx == 0:
            normalized_centroids[i] = contact_centroids[0]
            normalized_frames[i] = contact_frames[0]
            if normalized_tracks_3d is not None:
                normalized_tracks_3d[i] = tracks_3d[0]
            if normalized_left_frames is not None:
                normalized_left_frames[i] = left_contact_frames[0]
            if normalized_right_frames is not None:
                normalized_right_frames[i] = right_contact_frames[0]
        elif idx >= T:
            normalized_centroids[i] = contact_centroids[-1]
            normalized_frames[i] = contact_frames[-1]
            if normalized_tracks_3d is not None:
                normalized_tracks_3d[i] = tracks_3d[-1]
            if normalized_left_frames is not None:
                normalized_left_frames[i] = left_contact_frames[-1]
            if normalized_right_frames is not None:
                normalized_right_frames[i] = right_contact_frames[-1]
        else:
            # Linear interpolation between frames idx-1 and idx
            d0 = cumulative_distance[idx - 1]
            d1 = cumulative_distance[idx]
            if d1 - d0 < 1e-10:
                t = 0.0
            else:
                t = (target_dist - d0) / (d1 - d0)

            # Interpolate centroid (position)
            normalized_centroids[i] = (1 - t) * contact_centroids[idx - 1] + t * contact_centroids[idx]

            # Interpolate frame (position + rotation via SLERP)
            normalized_frames[i] = _interpolate_transform(
                contact_frames[idx - 1], contact_frames[idx], t
            )

            # Interpolate track points
            if normalized_tracks_3d is not None:
                normalized_tracks_3d[i] = (1 - t) * tracks_3d[idx - 1] + t * tracks_3d[idx]

            # Interpolate per-finger frames
            if normalized_left_frames is not None:
                normalized_left_frames[i] = _interpolate_transform(
                    left_contact_frames[idx - 1], left_contact_frames[idx], t
                )
            if normalized_right_frames is not None:
                normalized_right_frames[i] = _interpolate_transform(
                    right_contact_frames[idx - 1], right_contact_frames[idx], t
                )

    # Map original frames to nearest normalized index
    frame_to_normalized_idx = np.zeros(T, dtype=np.int32)
    for i in range(T):
        # Find closest target distance
        frame_to_normalized_idx[i] = np.argmin(np.abs(target_distances - cumulative_distance[i]))

    # Build result dict
    result['normalized_centroids'] = normalized_centroids
    result['normalized_frames'] = normalized_frames
    result['cumulative_distance_mm'] = cumulative_distance * 1000  # Return in mm
    result['frame_to_normalized_idx'] = frame_to_normalized_idx
    result['num_normalized_steps'] = num_steps

    if normalized_tracks_3d is not None:
        result['normalized_tracks_3d'] = normalized_tracks_3d
    if normalized_left_frames is not None:
        result['normalized_left_frames'] = normalized_left_frames
    if normalized_right_frames is not None:
        result['normalized_right_frames'] = normalized_right_frames

    return result


def _interpolate_transform(T0, T1, t):
    """Interpolate between two 4x4 transforms using linear + SLERP.

    Args:
        T0: [4, 4] start transform
        T1: [4, 4] end transform
        t: interpolation parameter [0, 1]

    Returns:
        T_interp: [4, 4] interpolated transform
    """
    # Interpolate position linearly
    pos = (1 - t) * T0[:3, 3] + t * T1[:3, 3]

    # Interpolate rotation using SLERP
    R0 = R.from_matrix(T0[:3, :3])
    R1 = R.from_matrix(T1[:3, :3])

    # SLERP via quaternions
    q0 = R0.as_quat()
    q1 = R1.as_quat()

    # Ensure shortest path
    if np.dot(q0, q1) < 0:
        q1 = -q1

    # Spherical linear interpolation
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot > 0.9995:
        # Very close - use linear interpolation
        q_interp = (1 - t) * q0 + t * q1
        q_interp = q_interp / np.linalg.norm(q_interp)
    else:
        theta = np.arccos(dot)
        q_interp = (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)

    R_interp = R.from_quat(q_interp).as_matrix()

    # Build output transform
    T_interp = np.eye(4, dtype=np.float32)
    T_interp[:3, :3] = R_interp
    T_interp[:3, 3] = pos

    return T_interp


class ContactSurfaceTracker:
    """Sample and track contact surface points on the gripper fingers."""

    def __init__(self, num_track_points=24, mesh_path=DEFAULT_INNER_FINGER_MESH):
        self.num_track_points = num_track_points
        self.mesh_path = mesh_path
        self.contact_points_local = None
        self.contact_mesh_vertices = None
        self.contact_mesh_faces = None

        if os.path.exists(mesh_path):
            finger_mesh = trimesh.load(mesh_path)
            contact_mesh = extract_contact_surface(finger_mesh)
            if contact_mesh is not None:
                self.contact_mesh_vertices = contact_mesh.vertices.copy()
                self.contact_mesh_faces = contact_mesh.faces.copy()
                self.contact_points_local = sample_contact_points(contact_mesh, num_track_points)
        else:
            print(f"[WARN] Mesh not found for contact sampling: {mesh_path}")

    def get_contact_points_world(self, T_base_ee, gripper_pos):
        """Return sampled contact surface points for both fingers in world frame."""
        if self.contact_points_local is None:
            return None, None

        pts_local = np.hstack([self.contact_points_local, np.ones((len(self.contact_points_local), 1))])
        T_left, T_right = compute_finger_transforms(T_base_ee, gripper_pos)

        pts_left_world = (T_left @ pts_local.T).T[:, :3]
        pts_right_world = (T_right @ pts_local.T).T[:, :3]
        return pts_left_world, pts_right_world

    def get_contact_points_and_frames(self, T_base_ee, gripper_pos):
        """Return contact points AND contact frames for both fingers.

        Returns:
            pts_left_world: [N, 3] left finger contact points in world frame
            pts_right_world: [N, 3] right finger contact points in world frame
            contact_centroid: [3] center of mass of all contact points
            contact_frame: [4, 4] combined contact frame (centroid + EE orientation)
            left_contact_frame: [4, 4] left finger contact frame
            right_contact_frame: [4, 4] right finger contact frame
        """
        if self.contact_points_local is None:
            return None, None, None, None, None, None

        pts_local = np.hstack([self.contact_points_local, np.ones((len(self.contact_points_local), 1))])
        T_left, T_right = compute_finger_transforms(T_base_ee, gripper_pos)

        pts_left_world = (T_left @ pts_local.T).T[:, :3]
        pts_right_world = (T_right @ pts_local.T).T[:, :3]

        # Compute combined contact frame (center of mass of all points)
        contact_centroid, contact_frame = compute_contact_frame(
            pts_left_world, pts_right_world, T_base_ee
        )

        # Compute per-finger contact frames
        _, left_contact_frame = compute_contact_frame_per_finger(pts_left_world, T_left)
        _, right_contact_frame = compute_contact_frame_per_finger(pts_right_world, T_right)

        return (pts_left_world, pts_right_world,
                contact_centroid, contact_frame,
                left_contact_frame, right_contact_frame)


class MinimalGripperVisualizer:
    """Simplified gripper visualizer that renders fingers with highlighted contact surfaces."""

    def __init__(self, root_path="world/gripper", num_track_points=50, mesh_path=DEFAULT_INNER_FINGER_MESH):
        self.root_path = root_path
        self.mesh_path = mesh_path
        # Gripper color configuration
        self.gripper_color = [255, 255, 255]  # White for the finger body
        self.contact_color = [70, 130, 180]   # Blue for contact surface

        self.finger_mesh = None
        self.contact_mesh = None
        self.num_track_points = num_track_points
        self.contact_points_local = None  # Points sampled from contact surface in local frame

        if os.path.exists(mesh_path):
            self.finger_mesh = trimesh.load(mesh_path)
            # Extract the full contact surface (all faces facing -Y direction)
            self.contact_mesh = extract_contact_surface(self.finger_mesh)
            # Sample points from contact surface for tracking
            if self.contact_mesh is not None:
                self.contact_points_local = sample_contact_points(self.contact_mesh, num_track_points)
        else:
            print(f"[WARN] Mesh not found: {mesh_path}")

    def init_rerun(self):
        """Initialize Rerun visualization with finger meshes and contact surfaces."""
        # Log the full white finger meshes
        if self.finger_mesh is not None:
            num_vertices = len(self.finger_mesh.vertices)
            vertex_colors = np.tile(self.gripper_color, (num_vertices, 1)).astype(np.uint8)

            rr.log(
                f"{self.root_path}/left_inner_knuckle/left_inner_finger/mesh",
                rr.Mesh3D(
                    vertex_positions=self.finger_mesh.vertices,
                    vertex_normals=self.finger_mesh.vertex_normals,
                    vertex_colors=vertex_colors,
                    triangle_indices=self.finger_mesh.faces
                ),
                static=True
            )
            rr.log(
                f"{self.root_path}/right_inner_knuckle/right_inner_finger/mesh",
                rr.Mesh3D(
                    vertex_positions=self.finger_mesh.vertices,
                    vertex_normals=self.finger_mesh.vertex_normals,
                    vertex_colors=vertex_colors,
                    triangle_indices=self.finger_mesh.faces
                ),
                static=True
            )

        # Log the blue contact surfaces (overlaid on the fingers)
        if self.contact_mesh is not None:
            num_vertices = len(self.contact_mesh.vertices)
            vertex_colors = np.tile(self.contact_color, (num_vertices, 1)).astype(np.uint8)

            rr.log(
                f"{self.root_path}/left_inner_knuckle/left_inner_finger/contact",
                rr.Mesh3D(
                    vertex_positions=self.contact_mesh.vertices,
                    vertex_normals=self.contact_mesh.vertex_normals,
                    vertex_colors=vertex_colors,
                    triangle_indices=self.contact_mesh.faces
                ),
                static=True
            )
            rr.log(
                f"{self.root_path}/right_inner_knuckle/right_inner_finger/contact",
                rr.Mesh3D(
                    vertex_positions=self.contact_mesh.vertices,
                    vertex_normals=self.contact_mesh.vertex_normals,
                    vertex_colors=vertex_colors,
                    triangle_indices=self.contact_mesh.faces
                ),
                static=True
            )

    def get_contact_points_world(self, T_base_ee, gripper_pos):
        """Get contact surface points in world coordinates for both fingers."""
        if self.contact_points_local is None:
            return None, None

        T_left, T_right = compute_finger_transforms(T_base_ee, gripper_pos)

        # Transform local points to world space
        pts_local = np.hstack([self.contact_points_local, np.ones((len(self.contact_points_local), 1))])

        pts_left_world = (T_left @ pts_local.T).T[:, :3]
        pts_right_world = (T_right @ pts_local.T).T[:, :3]

        return pts_left_world, pts_right_world

    def update(self, T_base_ee, gripper_pos):
        """Update only the two final finger transforms."""
        # Update root transform
        rr.log(
            self.root_path,
            rr.Transform3D(
                translation=T_base_ee[:3, 3],
                mat3x3=T_base_ee[:3, :3],
                axis_length=0.1
            )
        )

        # Calculate joint angle
        val = gripper_pos[0] if isinstance(gripper_pos, (list, np.ndarray)) else gripper_pos
        theta = val * 0.8

        # Left inner knuckle transform
        T_lik = np.eye(4)
        T_lik[:3, 3] = [0.0127, 0, 0.0693]
        T_lik[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
        rr.log(
            f"{self.root_path}/left_inner_knuckle",
            rr.Transform3D(translation=T_lik[:3, 3], mat3x3=T_lik[:3, :3])
        )

        # Left inner finger transform
        T_lif = np.eye(4)
        T_lif[:3, 3] = [0.03458531, 0, 0.04549702]
        T_lif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        rr.log(
            f"{self.root_path}/left_inner_knuckle/left_inner_finger",
            rr.Transform3D(translation=T_lif[:3, 3], mat3x3=T_lif[:3, :3])
        )

        # Right inner knuckle transform
        T_rik_origin = np.eye(4)
        T_rik_origin[:3, 3] = [-0.0127, 0, 0.0693]
        T_rik_origin[:3, :3] = R.from_euler('z', np.pi).as_matrix()
        R_rik_joint = R.from_rotvec([0, -theta, 0]).as_matrix()
        T_rik = T_rik_origin.copy()
        T_rik[:3, :3] = T_rik[:3, :3] @ R_rik_joint
        rr.log(
            f"{self.root_path}/right_inner_knuckle",
            rr.Transform3D(translation=T_rik[:3, 3], mat3x3=T_rik[:3, :3])
        )

        # Right inner finger transform
        T_rif = np.eye(4)
        T_rif[:3, 3] = [0.03410605, 0, 0.04585739]
        T_rif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        rr.log(
            f"{self.root_path}/right_inner_knuckle/right_inner_finger",
            rr.Transform3D(translation=T_rif[:3, 3], mat3x3=T_rif[:3, :3])
        )
