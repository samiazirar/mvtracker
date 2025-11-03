"""
Batch convert multiple BEHAVE scenes to .npz format.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def get_available_scenes(behave_root: Path, pattern: Optional[str] = None) -> List[str]:
    """Get list of available scenes in BEHAVE dataset."""
    scenes = []
    
    for scene_path in sorted(behave_root.iterdir()):
        if scene_path.is_dir() and not scene_path.name.startswith('.'):
            # Check if it has the expected structure (info.json and timestamp folders)
            info_file = scene_path / "info.json"
            if info_file.exists():
                if pattern is None or pattern in scene_path.name:
                    scenes.append(scene_path.name)
    
    return scenes


def convert_scene(
    scene: str,
    behave_root: str,
    output_dir: str,
    mask_type: str,
    downscale_factor: int,
    max_frames: Optional[int] = None,
    save_rerun: bool = False,
) -> bool:
    """Convert a single scene."""
    
    cmd = [
        sys.executable,
        "conversions/behave_to_npz.py",
        "--behave_root", behave_root,
        "--scene", scene,
        "--output_dir", output_dir,
        "--mask_type", mask_type,
        "--downscale_factor", str(downscale_factor),
    ]
    
    if max_frames is not None:
        cmd.extend(["--max_frames", str(max_frames)])
    
    if save_rerun:
        cmd.append("--save_rerun")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error converting {scene}: {e}")
        return False


def inspect_scene(npz_path: Path) -> bool:
    """Inspect a converted scene."""
    
    cmd = [
        sys.executable,
        "conversions/inspect_behave_npz.py",
        str(npz_path),
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error inspecting {npz_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch convert BEHAVE scenes to .npz format")
    parser.add_argument(
        "--behave_root",
        type=str,
        default="/data/behave-dataset/behave_all",
        help="Path to BEHAVE dataset root"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./conversions/behave_converted",
        help="Output directory for .npz files"
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=None,
        help="List of scene names to convert (default: all scenes)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Pattern to filter scenes (e.g., 'Date01' or 'basketball')"
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="person",
        choices=["person", "hand"],
        help="Type of mask to use"
    )
    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=2,
        help="Factor to downscale images"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum number of frames per scene (None for all)"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect each converted file after conversion"
    )
    parser.add_argument(
        "--save_rerun",
        action="store_true",
        help="Save rerun visualization for each scene"
    )
    parser.add_argument(
        "--list_scenes",
        action="store_true",
        help="List available scenes and exit"
    )
    
    args = parser.parse_args()
    
    behave_root = Path(args.behave_root)
    output_dir = Path(args.output_dir)
    
    if not behave_root.exists():
        print(f"Error: BEHAVE root not found: {behave_root}")
        return
    
    # Get list of scenes
    if args.scenes is not None:
        scenes = args.scenes
    else:
        scenes = get_available_scenes(behave_root, args.pattern)
    
    if args.list_scenes:
        print(f"Available scenes in {behave_root}:")
        for scene in scenes:
            print(f"  - {scene}")
        print(f"\nTotal: {len(scenes)} scenes")
        return
    
    if not scenes:
        print("No scenes found to convert")
        return
    
    print(f"Found {len(scenes)} scenes to convert")
    print(f"Output directory: {output_dir}")
    print("")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each scene
    success_count = 0
    failed_scenes = []
    
    for i, scene in enumerate(scenes, 1):
        print("=" * 70)
        print(f"Converting scene {i}/{len(scenes)}: {scene}")
        print("=" * 70)
        
        success = convert_scene(
            scene=scene,
            behave_root=str(behave_root),
            output_dir=str(output_dir),
            mask_type=args.mask_type,
            downscale_factor=args.downscale_factor,
            max_frames=args.max_frames,
            save_rerun=args.save_rerun,
        )
        
        if success:
            print(f"✓ Successfully converted: {scene}")
            success_count += 1
            
            # Optionally inspect
            if args.inspect:
                print(f"Inspecting converted file...")
                npz_path = output_dir / f"{scene}.npz"
                inspect_scene(npz_path)
        else:
            print(f"✗ Failed to convert: {scene}")
            failed_scenes.append(scene)
        
        print("")
    
    # Summary
    print("=" * 70)
    print("Batch conversion complete!")
    print("=" * 70)
    print(f"Successfully converted: {success_count}/{len(scenes)} scenes")
    
    if failed_scenes:
        print(f"\nFailed scenes:")
        for scene in failed_scenes:
            print(f"  - {scene}")
    
    print(f"\nOutput directory: {output_dir}")
    
    # List converted files
    npz_files = sorted(output_dir.glob("*.npz"))
    if npz_files:
        print(f"\nConverted files ({len(npz_files)}):")
        total_size = 0
        for npz_file in npz_files:
            size_mb = npz_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  - {npz_file.name}: {size_mb:.2f} MB")
        print(f"\nTotal size: {total_size:.2f} MB")


if __name__ == "__main__":
    main()
