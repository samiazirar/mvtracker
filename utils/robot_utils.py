#!/usr/bin/env python3
"""
Robot model and manipulation utilities for RH20T dataset.

This module provides functions for working with robot models including:
- _create_robot_model: Initialize RobotModel from URDF and mesh files
- _load_urdf_link_colors: Extract material colors from URDF file
- _load_mtl_colors_for_mesh_dir: Load material colors from MTL files
- _robot_model_to_pointcloud: Convert robot state to colored point cloud
"""

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from RH20T.rh20t_api.configurations import get_conf_from_dir_name, load_conf
from RH20T.utils.robot import RobotModel


def _create_robot_model(
    sample_path: Path,
    configs_path: Optional[Path] = None,
    rh20t_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Create a RobotModel instance for the current scene."""
    configs_path = configs_path or (Path(__file__).resolve().parent / "rh20t_api" / "configs" / "configs.json")
    rh20t_root = rh20t_root or (Path(__file__).resolve().parent / "rh20t_api")

    try:
        confs = load_conf(str(configs_path))
    except Exception as exc:  # pragma: no cover - best-effort logging
        warnings.warn(f"Failed to load RH20T configurations ({exc}); skipping robot overlay.")
        return None

    try:
        conf = get_conf_from_dir_name(str(sample_path), confs)
    except Exception as exc:  # pragma: no cover - dataset naming mismatches
        warnings.warn(f"Could not infer robot configuration from '{sample_path}' ({exc}); skipping robot overlay.")
        return None

    robot_urdf = (rh20t_root / conf.robot_urdf).resolve()
    robot_mesh_root = (rh20t_root / conf.robot_mesh).resolve()

    if not robot_urdf.exists():
        warnings.warn(f"Robot URDF not found at '{robot_urdf}'; skipping robot overlay.")
        return None
    if not robot_mesh_root.exists():
        warnings.warn(f"Robot mesh directory '{robot_mesh_root}' not found; skipping robot overlay.")
        return None

    try:
        robot_model = RobotModel(
            robot_joint_sequence=conf.robot_joint_sequence,
            robot_urdf=str(robot_urdf),
            robot_mesh=str(robot_mesh_root),
        )
    except Exception as exc:  # pragma: no cover - URDF parsing is best-effort
        warnings.warn(f"Failed to load robot model from URDF ({exc}); skipping robot overlay.")
        return None
    
    # Load MTL colors from mesh directory (graphics subdirectory)
    graphics_dir = robot_mesh_root / "graphics"
    if not graphics_dir.exists():
        graphics_dir = robot_mesh_root  # Fallback to robot_mesh_root itself
    mtl_colors = _load_mtl_colors_for_mesh_dir(graphics_dir)

    return {
        "model": robot_model,
        "mtl_colors": mtl_colors,
    }


def _load_urdf_link_colors(urdf_path: Path) -> Dict[str, np.ndarray]:
    """Parse URDF file to extract material colors for each link.
    
    Returns:
        Dictionary mapping link names to RGB colors (0-1 range)
    """
    link_colors = {}
    
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        
        # First, collect material definitions
        materials = {}
        for material in root.findall('.//material'):
            name = material.get('name')
            if not name:
                continue
            color_elem = material.find('color')
            if color_elem is not None:
                rgba = color_elem.get('rgba')
                if rgba:
                    rgba_values = [float(x) for x in rgba.split()]
                    materials[name] = np.array(rgba_values[:3])  # RGB only
        
        # Now map links to their visual materials
        for link in root.findall('.//link'):
            link_name = link.get('name')
            if not link_name:
                continue
                
            visuals = link.findall('visual')
            if visuals:
                # Use first visual's material
                material = visuals[0].find('material')
                if material is not None:
                    mat_name = material.get('name')
                    color_elem = material.find('color')
                    
                    if color_elem is not None:
                        # Inline color definition
                        rgba = color_elem.get('rgba')
                        if rgba:
                            rgba_values = [float(x) for x in rgba.split()]
                            link_colors[link_name] = np.array(rgba_values[:3])
                    elif mat_name and mat_name in materials:
                        # Reference to material definition
                        link_colors[link_name] = materials[mat_name]
    
    except Exception as e:
        print(f"[WARN] Could not parse URDF colors: {e}")
    
    return link_colors


def _load_mtl_colors_for_mesh_dir(mesh_dir: Path) -> Dict[str, np.ndarray]:
    """Load material colors from MTL files in the mesh directory.
    
    Args:
        mesh_dir: Directory containing mesh and MTL files
        
    Returns:
        Dictionary mapping mesh filenames (without extension) to RGB colors (0-1 range)
    """
    mesh_colors = {}
    
    try:
        # Find all MTL files in the directory
        for mtl_file in mesh_dir.glob("*.mtl"):
            # Parse the MTL file
            material_color = None
            with open(mtl_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Look for diffuse color (Kd)
                    if line.startswith('Kd '):
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                r, g, b = float(parts[1]), float(parts[2]), float(parts[3])
                                material_color = np.array([r, g, b])
                                break
                            except ValueError:
                                continue
            
            if material_color is not None:
                # Map this color to all mesh files with the same base name
                base_name = mtl_file.stem  # e.g., "link0_body"
                mesh_colors[base_name] = material_color
                
    except Exception as e:
        print(f"[WARN] Could not parse MTL colors: {e}")
    
    return mesh_colors


def _robot_model_to_pointcloud(
    robot_model: RobotModel,
    joint_angles: np.ndarray,
    is_first_time: bool = False,
    points_per_mesh: int = 10000,
    debug_mode: bool = False,
    urdf_colors: Optional[Dict[str, np.ndarray]] = None,
    mtl_colors: Optional[Dict[str, np.ndarray]] = None,
) -> o3d.geometry.PointCloud:
    """Convert a robot model at a specific joint state to a point cloud."""
    # Update the robot model with the current joint angles
    # This transforms the internal meshes to the correct pose
    try:
        # Ensure joint_angles is a numpy array and convert to list for robot_model.update()
        if isinstance(joint_angles, np.ndarray):
            joint_angles_list = joint_angles.tolist()
        else:
            joint_angles_list = list(joint_angles)
        robot_model.update(joint_angles_list, first_time=is_first_time)
    except Exception as e:
        print(f"[ERROR] robot_model.update() failed: {e}")
        import traceback
        traceback.print_exc()
        return o3d.geometry.PointCloud()

    fk = None
    joint_sequence = getattr(robot_model, "joint_sequence", [])
    if joint_sequence:
        # Ensure joint_angles is array-like for indexing
        joint_angles_arr = np.asarray(joint_angles) if not isinstance(joint_angles, np.ndarray) else joint_angles
        if len(joint_sequence) != len(joint_angles_arr):
            warnings.warn(
                f"Joint angle length ({len(joint_angles_arr)}) does not match robot joint sequence ({len(joint_sequence)})."
            )
        try:
            joint_map = {name: float(joint_angles_arr[i]) for i, name in enumerate(joint_sequence)}
            fk = robot_model.chain.forward_kinematics(joint_map)
            # robot_model.latest_transforms = fk  # ← TODO: Check this line
        except Exception as exc:
            print(f"[ERROR] Forward kinematics computation failed: {exc}")
            warnings.warn(f"Failed to compute forward kinematics for robot model ({exc}).")
    
    # Combine all robot meshes into a single point cloud
    robot_pcd = o3d.geometry.PointCloud()
    try:
        geometries = robot_model.geometries_to_add if is_first_time else robot_model.geometries_to_update
    except Exception as e:
        print(f"[ERROR] Failed to get geometries: {e}")
        return robot_pcd
    
    # Define debug colors for different robot parts (only used if debug_mode=True)
    debug_part_colors = [
        [0.2, 0.3, 0.8],   # Blue
        [0.8, 0.3, 0.2],   # Red-orange
        [0.2, 0.8, 0.3],   # Green
        [0.8, 0.8, 0.2],   # Yellow
        [0.8, 0.2, 0.8],   # Magenta
        [0.2, 0.8, 0.8],   # Cyan
        [0.9, 0.5, 0.1],   # Orange
    ]
    
    for mesh_idx, mesh in enumerate(geometries):
        # The mesh vertices are already in the correct world position
        # thanks to robot_model.update() transforming them
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        if vertices.size == 0 or triangles.size == 0:
            continue

        # Create an Open3D mesh from the current transformed state
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # IMPORTANT: Copy vertex colors from original mesh if they exist
        if mesh.has_vertex_colors():
            o3d_mesh.vertex_colors = mesh.vertex_colors
        
        # Sample points from the mesh surface (already in correct position)
        mesh_pcd = o3d_mesh.sample_points_uniformly(number_of_points=points_per_mesh)
        
        # Assign colors based on mode
        if debug_mode:
            # Debug mode: Use bright, distinct colors for each part
            num_points = len(mesh_pcd.points)
            part_color = debug_part_colors[mesh_idx % len(debug_part_colors)]
            mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile(part_color, (num_points, 1)))
        else:
            # Normal mode: Try MTL colors first, then URDF colors, then mesh colors, then default gray
            color_assigned = False
            
            # First try: MTL material colors (most accurate colors from the actual mesh files)
            if mtl_colors:
                mesh_basename = robot_model.get_filename_for_mesh(mesh)
                if mesh_basename and mesh_basename in mtl_colors:
                    num_points = len(mesh_pcd.points)
                    mtl_color = mtl_colors[mesh_basename]
                    mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile(mtl_color, (num_points, 1)))
                    color_assigned = True
            
            # Second try: URDF material colors
            if not color_assigned and urdf_colors:
                link_name = robot_model.get_link_for_mesh(mesh)
                if link_name and link_name in urdf_colors:
                    num_points = len(mesh_pcd.points)
                    urdf_color = urdf_colors[link_name]
                    mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile(urdf_color, (num_points, 1)))
                    color_assigned = True
            
            # Third try: Original mesh vertex colors
            if not color_assigned and mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors)
                if colors.size > 0:
                    # Transfer colors from mesh to sampled points using nearest neighbor
                    kdtree = o3d.geometry.KDTreeFlann(o3d_mesh)
                    sampled_colors = np.zeros((len(mesh_pcd.points), 3))
                    for i, point in enumerate(np.asarray(mesh_pcd.points)):
                        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
                        sampled_colors[i] = colors[idx[0]]
                    mesh_pcd.colors = o3d.utility.Vector3dVector(sampled_colors)
                    color_assigned = True
            
            # Fallback: Use a neutral gray/silver color
            if not color_assigned:
                num_points = len(mesh_pcd.points)
                default_color = [0.7, 0.7, 0.7]  # Light gray
                mesh_pcd.colors = o3d.utility.Vector3dVector(np.tile(default_color, (num_points, 1)))
        
        robot_pcd += mesh_pcd

    return robot_pcd
