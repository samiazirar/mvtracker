"""Gripper visualization utilities for DROID point cloud generation."""

import numpy as np
import os
import trimesh
from scipy.spatial.transform import Rotation as R
import rerun as rr


class GripperVisualizer:
    """
    Visualizer for Robotiq 85 gripper using Rerun.
    
    Handles loading gripper meshes and updating joint transforms based on
    gripper position commands.
    """
    
    def __init__(self, root_path="world/gripper"):
        """
        Initialize gripper visualizer.
        
        Args:
            root_path: Root path in Rerun hierarchy for gripper (default: "world/gripper")
        """
        self.root_path = root_path
        self.meshes = {}
        # Paths relative to workspace root
        base_path = "/workspace/third_party/robotiq_arg85_description/meshes"
        self.mesh_files = {
            "base": f"{base_path}/robotiq_85_base_link_fine.STL",
            "outer_knuckle": f"{base_path}/outer_knuckle_fine.STL",
            "outer_finger": f"{base_path}/outer_finger_fine.STL",
            "inner_knuckle": f"{base_path}/inner_knuckle_fine.STL",
            "inner_finger": f"{base_path}/inner_finger_fine.STL",
        }
        self._load_meshes()

    def _load_meshes(self):
        """Load all gripper mesh files from disk."""
        for name, path in self.mesh_files.items():
            if os.path.exists(path):
                self.meshes[name] = trimesh.load(path)
            else:
                print(f"[WARN] Mesh not found: {path}")

    def init_rerun(self):
        """
        Initialize Rerun visualization by logging static meshes.
        
        Logs all gripper component meshes as static entities in the Rerun hierarchy.
        """
        # Log meshes once as static (relative to their parent transforms)
        # Base
        self._log_mesh("base", f"{self.root_path}/base/mesh")
        
        # Left Finger
        self._log_mesh("outer_knuckle", f"{self.root_path}/left_outer_knuckle/mesh")
        self._log_mesh("outer_finger", f"{self.root_path}/left_outer_knuckle/left_outer_finger/mesh")
        self._log_mesh("inner_knuckle", f"{self.root_path}/left_inner_knuckle/mesh")
        self._log_mesh("inner_finger", f"{self.root_path}/left_inner_knuckle/left_inner_finger/mesh")

        # Right Finger
        self._log_mesh("outer_knuckle", f"{self.root_path}/right_outer_knuckle/mesh")
        self._log_mesh("outer_finger", f"{self.root_path}/right_outer_knuckle/right_outer_finger/mesh")
        self._log_mesh("inner_knuckle", f"{self.root_path}/right_inner_knuckle/mesh")
        self._log_mesh("inner_finger", f"{self.root_path}/right_inner_knuckle/right_inner_finger/mesh")

    def _log_mesh(self, name, path):
        """
        Log a single mesh to Rerun.
        
        Args:
            name: Name key in self.meshes dictionary
            path: Rerun entity path for the mesh
        """
        if name in self.meshes:
            mesh = self.meshes[name]
            rr.log(path, rr.Mesh3D(vertex_positions=mesh.vertices, vertex_normals=mesh.vertex_normals, triangle_indices=mesh.faces), static=True)

    def update(self, T_base_ee, gripper_pos):
        """
        Update gripper visualization with new pose and joint state.
        
        Args:
            T_base_ee: 4x4 transformation matrix from base to end-effector
            gripper_pos: Gripper position command (scalar or array)
        """
        # 1. Update Root (End Effector)
        rr.log(self.root_path, rr.Transform3D(translation=T_base_ee[:3, 3], mat3x3=T_base_ee[:3, :3], axis_length=0.1))
        
        # 2. Calculate Joint Angles
        val = gripper_pos[0] if isinstance(gripper_pos, (list, np.ndarray)) else gripper_pos
        theta = val * 0.8 

        # 3. Update Parts Transforms
        
        # Left Outer Knuckle
        # Origin: 0.03060114 0 0.06279202
        # Axis: 0 -1 0 -> Rot: [0, -theta, 0]
        T_lok = np.eye(4)
        T_lok[:3, 3] = [0.03060114, 0, 0.06279202]
        T_lok[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
        rr.log(f"{self.root_path}/left_outer_knuckle", rr.Transform3D(translation=T_lok[:3, 3], mat3x3=T_lok[:3, :3]))

        # Left Outer Finger (Fixed relative to knuckle)
        # Origin: 0.03169104 0 -0.00193396
        T_lof = np.eye(4)
        T_lof[:3, 3] = [0.03169104, 0, -0.00193396]
        rr.log(f"{self.root_path}/left_outer_knuckle/left_outer_finger", rr.Transform3D(translation=T_lof[:3, 3]))

        # Left Inner Knuckle
        # Origin: 0.0127 0 0.0693
        # Axis: 0 -1 0 -> Rot: [0, -theta, 0] (Mimic 1)
        T_lik = np.eye(4)
        T_lik[:3, 3] = [0.0127, 0, 0.0693]
        T_lik[:3, :3] = R.from_rotvec([0, -theta, 0]).as_matrix()
        rr.log(f"{self.root_path}/left_inner_knuckle", rr.Transform3D(translation=T_lik[:3, 3], mat3x3=T_lik[:3, :3]))

        # Left Inner Finger
        # Origin: 0.03458531 0 0.04549702
        # Axis: 0 -1 0 -> Rot: [0, theta, 0] (Mimic -1)
        T_lif = np.eye(4)
        T_lif[:3, 3] = [0.03458531, 0, 0.04549702]
        T_lif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        rr.log(f"{self.root_path}/left_inner_knuckle/left_inner_finger", rr.Transform3D(translation=T_lif[:3, 3], mat3x3=T_lif[:3, :3]))

        # Right Outer Knuckle
        # Origin: -0.03060114 0 0.06279202
        # Base Rot: Z 180
        # Axis: 0 1 0 -> Rot: [0, -theta, 0] (Mimic -1)
        T_rok_origin = np.eye(4)
        T_rok_origin[:3, 3] = [-0.03060114, 0, 0.06279202]
        T_rok_origin[:3, :3] = R.from_euler('z', np.pi).as_matrix()
        
        R_rok_joint = R.from_rotvec([0, -theta, 0]).as_matrix()
        T_rok = T_rok_origin.copy()
        T_rok[:3, :3] = T_rok[:3, :3] @ R_rok_joint
        
        rr.log(f"{self.root_path}/right_outer_knuckle", rr.Transform3D(translation=T_rok[:3, 3], mat3x3=T_rok[:3, :3]))

        # Right Outer Finger (Fixed)
        # Origin: 0.03169104 0 -0.00193396
        T_rof = np.eye(4)
        T_rof[:3, 3] = [0.03169104, 0, -0.00193396]
        rr.log(f"{self.root_path}/right_outer_knuckle/right_outer_finger", rr.Transform3D(translation=T_rof[:3, 3]))

        # Right Inner Knuckle
        # Origin: -0.0127 0 0.0693
        # Base Rot: Z 180
        # Axis: 0 1 0 -> Rot: [0, -theta, 0] (Mimic -1)
        T_rik_origin = np.eye(4)
        T_rik_origin[:3, 3] = [-0.0127, 0, 0.0693]
        T_rik_origin[:3, :3] = R.from_euler('z', np.pi).as_matrix()
        
        R_rik_joint = R.from_rotvec([0, -theta, 0]).as_matrix()
        T_rik = T_rik_origin.copy()
        T_rik[:3, :3] = T_rik[:3, :3] @ R_rik_joint
        
        rr.log(f"{self.root_path}/right_inner_knuckle", rr.Transform3D(translation=T_rik[:3, 3], mat3x3=T_rik[:3, :3]))

        # Right Inner Finger
        # Origin: 0.03410605 0 0.04585739
        # Axis: 0 1 0 -> Rot: [0, theta, 0] (Mimic 1)
        T_rif = np.eye(4)
        T_rif[:3, 3] = [0.03410605, 0, 0.04585739]
        T_rif[:3, :3] = R.from_rotvec([0, theta, 0]).as_matrix()
        rr.log(f"{self.root_path}/right_inner_knuckle/right_inner_finger", rr.Transform3D(translation=T_rif[:3, 3], mat3x3=T_rif[:3, :3]))
