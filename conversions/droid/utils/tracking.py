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
    if len(mesh.vertices) <= num_points:
        points = mesh.vertices.copy()
    else:
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


class ContactSurfaceTracker:
    """Sample and track contact surface points on the gripper fingers."""

    def __init__(self, num_track_points=24, mesh_path=DEFAULT_INNER_FINGER_MESH):
        self.num_track_points = num_track_points
        self.mesh_path = mesh_path
        self.contact_points_local = None

        if os.path.exists(mesh_path):
            finger_mesh = trimesh.load(mesh_path)
            contact_mesh = extract_contact_surface(finger_mesh)
            if contact_mesh is not None:
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
