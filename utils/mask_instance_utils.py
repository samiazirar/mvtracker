#!/usr/bin/env python3
"""
Utilities for splitting and processing mask instances separately.

This module provides functions to:
- Extract individual mask instances from NPZ files
- Create per-instance NPZ files
- Get list of mask instance IDs
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def get_mask_instance_ids(npz_path: Path, mask_key: str = "sam2_masks") -> List[str]:
    """
    Get list of mask instance IDs from an NPZ file.
    
    Args:
        npz_path: Path to NPZ file containing masks
        mask_key: Key for masks in NPZ (default: "sam2_masks")
        
    Returns:
        List of instance ID strings (e.g., ["instance_0", "instance_1"])
        
    Raises:
        KeyError: If mask_key not found in NPZ
        ValueError: If masks are not in expected dict format
    """
    data = np.load(npz_path, allow_pickle=True)
    
    if mask_key not in data:
        raise KeyError(f"Key '{mask_key}' not found in NPZ. Available: {list(data.keys())}")
    
    masks = data[mask_key].item()
    
    if not isinstance(masks, dict):
        raise ValueError(f"Expected masks to be a dict, got {type(masks)}")
    
    instance_ids = sorted(masks.keys())
    
    print(f"[INFO] Found {len(instance_ids)} mask instances: {instance_ids}")
    
    return instance_ids


def create_single_instance_npz(
    input_npz_path: Path,
    instance_id: str,
    output_npz_path: Path,
    mask_key: str = "sam2_masks",
    keep_all_data: bool = True,
) -> Path:
    """
    Create a new NPZ file containing only a single mask instance.
    
    This extracts one instance from the mask dictionary and creates a new NPZ
    with that instance's data, suitable for independent tracking.
    
    Args:
        input_npz_path: Path to input NPZ with multiple instances
        instance_id: ID of instance to extract (e.g., "instance_0")
        output_npz_path: Path for output NPZ file
        mask_key: Key for masks in NPZ (default: "sam2_masks")
        keep_all_data: If True, copy all other NPZ data; if False, only essential data
        
    Returns:
        Path to created NPZ file
        
    Raises:
        KeyError: If mask_key or instance_id not found
    """
    print(f"[INFO] Extracting instance '{instance_id}' from {input_npz_path.name}")
    
    # Load data
    data = np.load(input_npz_path, allow_pickle=True)
    
    if mask_key not in data:
        raise KeyError(f"Key '{mask_key}' not found in NPZ. Available: {list(data.keys())}")
    
    masks = data[mask_key].item()
    
    if instance_id not in masks:
        raise KeyError(f"Instance '{instance_id}' not found in masks. Available: {list(masks.keys())}")
    
    # Create output payload
    if keep_all_data:
        # Copy all data
        payload = dict(data)
    else:
        # Copy only essential data
        essential_keys = [
            "rgbs", "depths", "intrs", "extrs", "camera_ids",
            "timestamps", "gripper_states", "robot_states",
        ]
        payload = {k: data[k] for k in essential_keys if k in data}
    
    # Replace mask dict with single instance
    payload[mask_key] = {instance_id: masks[instance_id]}
    
    # Also extract contact frames if present
    contact_key = mask_key.replace("_masks", "_contact_frames")
    if contact_key in data:
        contact_frames = data[contact_key].item()
        if instance_id in contact_frames:
            payload[contact_key] = {instance_id: contact_frames[instance_id]}
    
    # Add metadata
    payload["instance_id"] = instance_id
    payload["original_file"] = str(input_npz_path)
    
    # Save
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz_path, **payload)
    
    print(f"[INFO] Saved instance NPZ to {output_npz_path}")
    
    return output_npz_path


def split_npz_by_instances(
    input_npz_path: Path,
    output_dir: Path,
    mask_key: str = "sam2_masks",
    suffix: str = "",
) -> Dict[str, Path]:
    """
    Split an NPZ file into separate files, one per mask instance.
    
    Args:
        input_npz_path: Path to input NPZ with multiple instances
        output_dir: Directory to save individual instance NPZ files
        mask_key: Key for masks in NPZ (default: "sam2_masks")
        suffix: Optional suffix to add to output filenames (e.g., "_query")
        
    Returns:
        Dictionary mapping instance_id -> output_npz_path
        
    Example:
        >>> paths = split_npz_by_instances(
        ...     Path("data/scene_masks.npz"),
        ...     Path("data/instances/"),
        ...     mask_key="sam2_masks",
        ...     suffix="_query"
        ... )
        >>> # Creates: data/instances/scene_masks_instance_0_query.npz, etc.
    """
    print(f"\n[INFO] ========== Splitting NPZ by Instances ==========")
    print(f"[INFO] Input: {input_npz_path}")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Get all instance IDs
    instance_ids = get_mask_instance_ids(input_npz_path, mask_key=mask_key)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into individual files
    instance_paths = {}
    
    for instance_id in instance_ids:
        # Generate output filename
        output_name = f"{input_npz_path.stem}_{instance_id}{suffix}.npz"
        output_path = output_dir / output_name
        
        # Create single-instance NPZ
        create_single_instance_npz(
            input_npz_path=input_npz_path,
            instance_id=instance_id,
            output_npz_path=output_path,
            mask_key=mask_key,
            keep_all_data=True,
        )
        
        instance_paths[instance_id] = output_path
    
    print(f"\n[INFO] ========== Split Complete ==========")
    print(f"[INFO] Created {len(instance_paths)} instance NPZ files:")
    for instance_id, path in instance_paths.items():
        print(f"[INFO]   {instance_id}: {path.name}")
    
    return instance_paths


def verify_instance_npz(npz_path: Path, expected_instance_id: str, mask_key: str = "sam2_masks") -> bool:
    """
    Verify that an NPZ file contains exactly one mask instance with the expected ID.
    
    Args:
        npz_path: Path to NPZ file to verify
        expected_instance_id: Expected instance ID
        mask_key: Key for masks in NPZ
        
    Returns:
        True if valid, False otherwise
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        if mask_key not in data:
            print(f"[ERROR] Key '{mask_key}' not found in {npz_path.name}")
            return False
        
        masks = data[mask_key].item()
        
        if not isinstance(masks, dict):
            print(f"[ERROR] Masks not in dict format in {npz_path.name}")
            return False
        
        if len(masks) != 1:
            print(f"[ERROR] Expected 1 instance, found {len(masks)} in {npz_path.name}")
            return False
        
        actual_id = list(masks.keys())[0]
        if actual_id != expected_instance_id:
            print(f"[ERROR] Expected '{expected_instance_id}', found '{actual_id}' in {npz_path.name}")
            return False
        
        print(f"[INFO] ✓ Verified {npz_path.name}: contains '{expected_instance_id}'")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to verify {npz_path.name}: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split NPZ file by mask instances")
    parser.add_argument("--npz", type=Path, required=True, help="Input NPZ file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--mask-key", type=str, default="sam2_masks", help="Mask key in NPZ")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for output files")
    
    args = parser.parse_args()
    
    instance_paths = split_npz_by_instances(
        input_npz_path=args.npz,
        output_dir=args.output_dir,
        mask_key=args.mask_key,
        suffix=args.suffix,
    )
    
    print(f"\n[INFO] Done! Created {len(instance_paths)} files in {args.output_dir}")
