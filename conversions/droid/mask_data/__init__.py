"""
DROID Mask Data Generation Module

This module provides utilities for generating 2D masks for:
- Robotiq 85 gripper (with articulated joints)
- Panda robot arm (7-DOF with forward kinematics)

Usage:
    from mask_data import GripperMaskRenderer, RobotArmMaskRenderer
    
    # Render gripper mask
    gripper = GripperMaskRenderer()
    mask = gripper.render_mask(T_world_ee, gripper_pos, K, T_cam_world, width, height)
    
    # Render robot arm mask  
    robot = RobotArmMaskRenderer()
    mask = robot.render_mask(joint_angles, K, T_cam_world, width, height)
"""

from .mask_utils import (
    # Mesh loading
    load_meshes,
    load_gripper_meshes,
    load_robot_arm_meshes,
    
    # Forward kinematics
    pose6_to_T,
    make_transform,
    panda_forward_kinematics,
    robotiq_gripper_transforms,
    
    # Mesh projection
    transform_mesh,
    project_vertices_to_2d,
    render_mesh_mask,
    render_multiple_meshes_mask,
    
    # High-level renderers
    GripperMaskRenderer,
    RobotArmMaskRenderer,
    CombinedRobotMaskRenderer,
    
    # Paths
    GRIPPER_MESH_BASE,
    GRIPPER_MESH_BASE_ALT,
    ROBOT_MESH_BASE,
    GRIPPER_MESHES,
    ROBOT_ARM_MESHES,
)

__all__ = [
    # Mesh loading
    'load_meshes',
    'load_gripper_meshes',
    'load_robot_arm_meshes',
    
    # Forward kinematics
    'pose6_to_T',
    'make_transform',
    'panda_forward_kinematics',
    'robotiq_gripper_transforms',
    
    # Mesh projection  
    'transform_mesh',
    'project_vertices_to_2d',
    'render_mesh_mask',
    'render_multiple_meshes_mask',
    
    # High-level renderers
    'GripperMaskRenderer',
    'RobotArmMaskRenderer',
    'CombinedRobotMaskRenderer',
    
    # Paths
    'GRIPPER_MESH_BASE',
    'GRIPPER_MESH_BASE_ALT',
    'ROBOT_MESH_BASE',
    'GRIPPER_MESHES',
    'ROBOT_ARM_MESHES',
]
