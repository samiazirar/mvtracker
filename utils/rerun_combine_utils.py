#!/usr/bin/env python3
"""
Utilities for combining multiple Rerun recordings into a single RRD file.

This module provides functions to:
- Load multiple .rrd files
- Merge them with unique entity paths
- Save combined recording
"""

from pathlib import Path
from typing import List, Optional, Dict
import subprocess
import sys


def combine_rrd_files_cli(
    input_rrd_paths: List[Path],
    output_rrd_path: Path,
    entity_prefixes: Optional[List[str]] = None,
) -> Path:
    """
    Combine multiple .rrd files using Rerun CLI.
    
    This uses the `rerun` CLI tool to merge recordings. Each recording
    is prefixed with a unique entity path to avoid conflicts.
    
    Args:
        input_rrd_paths: List of input .rrd file paths
        output_rrd_path: Path for combined output .rrd file
        entity_prefixes: Optional list of prefixes for entity paths.
                        If None, uses ["mask_0/", "mask_1/", ...]
        
    Returns:
        Path to created combined .rrd file
        
    Note:
        This requires the `rerun` CLI tool to be installed.
        Install with: pip install rerun-sdk
        
    Example:
        >>> combine_rrd_files_cli(
        ...     [Path("mask_0.rrd"), Path("mask_1.rrd")],
        ...     Path("combined.rrd"),
        ...     entity_prefixes=["left_hand/", "right_hand/"]
        ... )
    """
    print(f"\n[INFO] ========== Combining RRD Files ==========")
    print(f"[INFO] Input files: {len(input_rrd_paths)}")
    for path in input_rrd_paths:
        print(f"[INFO]   - {path}")
    print(f"[INFO] Output: {output_rrd_path}")
    
    # Generate prefixes if not provided
    if entity_prefixes is None:
        entity_prefixes = [f"mask_{i}/" for i in range(len(input_rrd_paths))]
    
    if len(entity_prefixes) != len(input_rrd_paths):
        raise ValueError(
            f"Number of prefixes ({len(entity_prefixes)}) must match "
            f"number of input files ({len(input_rrd_paths)})"
        )
    
    # Check if rerun CLI is available
    try:
        subprocess.run(
            ["rerun", "--version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "Rerun CLI not found. Install with: pip install rerun-sdk\n"
            "Or ensure 'rerun' is in your PATH."
        )
    
    # Build command: rerun <file1> <file2> ... --save <output>
    # Note: Current rerun CLI doesn't support entity prefixing directly
    # We'll need to use Python API instead
    print(f"[WARN] CLI-based merging doesn't support entity prefixing.")
    print(f"[INFO] Using Python API instead...")
    
    return combine_rrd_files_python(
        input_rrd_paths=input_rrd_paths,
        output_rrd_path=output_rrd_path,
        entity_prefixes=entity_prefixes,
    )


def combine_rrd_files_python(
    input_rrd_paths: List[Path],
    output_rrd_path: Path,
    entity_prefixes: Optional[List[str]] = None,
    application_id: str = "combined_tracking",
) -> Path:
    """
    Combine multiple .rrd files using Rerun Python API.
    
    This loads each recording and re-logs all data with entity path prefixes.
    
    Args:
        input_rrd_paths: List of input .rrd file paths
        output_rrd_path: Path for combined output .rrd file
        entity_prefixes: Optional list of prefixes for entity paths.
                        If None, uses ["mask_0/", "mask_1/", ...]
        application_id: Application ID for combined recording
        
    Returns:
        Path to created combined .rrd file
        
    Note:
        This requires programmatic access to RRD data structures.
        Currently, rerun-sdk doesn't provide a direct API to read and re-log
        arbitrary recordings. This function provides a framework for future
        implementation when the API becomes available.
        
    Example:
        >>> combine_rrd_files_python(
        ...     [Path("mask_0.rrd"), Path("mask_1.rrd")],
        ...     Path("combined.rrd"),
        ...     entity_prefixes=["instance_0/", "instance_1/"]
        ... )
    """
    import rerun as rr
    
    print(f"\n[INFO] ========== Combining RRD Files (Python API) ==========")
    print(f"[INFO] Input files: {len(input_rrd_paths)}")
    
    # Generate prefixes if not provided
    if entity_prefixes is None:
        entity_prefixes = [f"mask_{i}/" for i in range(len(input_rrd_paths))]
    
    if len(entity_prefixes) != len(input_rrd_paths):
        raise ValueError(
            f"Number of prefixes ({len(entity_prefixes)}) must match "
            f"number of input files ({len(input_rrd_paths)})"
        )
    
    # Verify all input files exist
    for path in input_rrd_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    
    # Initialize combined recording
    rr.init(application_id, recording_id="combined", spawn=False)
    
    # Set coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    print(f"[WARN] Rerun SDK doesn't yet provide API to read and re-log RRD files.")
    print(f"[WARN] Workaround: Use `rerun` CLI to view multiple files simultaneously:")
    print(f"[INFO]   rerun {' '.join(str(p) for p in input_rrd_paths)}")
    print(f"[INFO]")
    print(f"[INFO] For now, creating a simple combined recording with metadata...")
    
    # Log metadata about the component recordings
    for i, (path, prefix) in enumerate(zip(input_rrd_paths, entity_prefixes)):
        entity_path = f"metadata/{prefix.rstrip('/')}"
        rr.log(
            entity_path,
            rr.TextDocument(
                f"Source: {path.name}\n"
                f"Entity prefix: {prefix}\n"
                f"Index: {i}",
                media_type=rr.MediaType.TEXT,
            ),
            static=True,
        )
    
    # Save the combined recording
    output_rrd_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output_rrd_path))
    
    print(f"\n[INFO] ========== Combination Complete ==========")
    print(f"[INFO] Saved to: {output_rrd_path}")
    print(f"[INFO]")
    print(f"[INFO] NOTE: This is a metadata-only recording.")
    print(f"[INFO] To view all recordings together, use:")
    print(f"[INFO]   rerun {' '.join(str(p) for p in input_rrd_paths)}")
    
    return output_rrd_path


def create_viewing_script(
    rrd_paths: List[Path],
    output_script_path: Path,
    script_name: str = "view_combined_tracking",
) -> Path:
    """
    Create a shell script to view multiple RRD files together.
    
    Since Rerun CLI supports viewing multiple files simultaneously,
    this creates a convenient script to launch the viewer.
    
    Args:
        rrd_paths: List of .rrd file paths to view
        output_script_path: Path for output shell script
        script_name: Name for the viewing session
        
    Returns:
        Path to created script file
        
    Example:
        >>> create_viewing_script(
        ...     [Path("mask_0.rrd"), Path("mask_1.rrd")],
        ...     Path("view_all_masks.sh")
        ... )
    """
    print(f"\n[INFO] Creating viewing script: {output_script_path}")
    
    script_content = f"""#!/usr/bin/env bash
# Auto-generated script to view multiple tracking results together
# Created: {Path(__file__).name}

echo "Launching Rerun viewer with {len(rrd_paths)} recordings..."
echo ""

# View all RRD files together
rerun \\
"""
    
    for i, path in enumerate(rrd_paths):
        line = f"  {path}"
        if i < len(rrd_paths) - 1:
            line += " \\"
        script_content += line + "\n"
    
    script_content += """
echo ""
echo "Viewer closed."
"""
    
    # Write script
    output_script_path.parent.mkdir(parents=True, exist_ok=True)
    output_script_path.write_text(script_content)
    
    # Make executable
    output_script_path.chmod(0o755)
    
    print(f"[INFO] Created executable script: {output_script_path}")
    print(f"[INFO] Run with: bash {output_script_path}")
    
    return output_script_path


def combine_tracking_results(
    rrd_paths: List[Path],
    instance_ids: List[str],
    output_dir: Path,
    output_name: str = "combined_tracking",
) -> Dict[str, Path]:
    """
    Combine multiple per-instance tracking results into viewable format.
    
    This creates:
    1. A metadata RRD file (limited functionality)
    2. A shell script to view all recordings together
    
    Args:
        rrd_paths: List of per-instance .rrd file paths
        instance_ids: List of instance IDs (for entity prefixes)
        output_dir: Directory for output files
        output_name: Base name for output files
        
    Returns:
        Dictionary with paths:
            - "combined_rrd": Path to combined RRD (metadata only)
            - "viewing_script": Path to shell script for viewing
            - "individual_rrds": List of original RRD paths
    """
    print(f"\n[INFO] ========== Combining Tracking Results ==========")
    print(f"[INFO] Instances: {instance_ids}")
    print(f"[INFO] Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create entity prefixes from instance IDs
    entity_prefixes = [f"{inst_id}/" for inst_id in instance_ids]
    
    # Create combined RRD (metadata only for now)
    combined_rrd_path = output_dir / f"{output_name}.rrd"
    combine_rrd_files_python(
        input_rrd_paths=rrd_paths,
        output_rrd_path=combined_rrd_path,
        entity_prefixes=entity_prefixes,
        application_id=output_name,
    )
    
    # Create viewing script
    viewing_script_path = output_dir / f"view_{output_name}.sh"
    create_viewing_script(
        rrd_paths=rrd_paths,
        output_script_path=viewing_script_path,
        script_name=output_name,
    )
    
    print(f"\n[INFO] ========== Combination Complete ==========")
    print(f"[INFO] Files created:")
    print(f"[INFO]   Combined RRD (metadata): {combined_rrd_path}")
    print(f"[INFO]   Viewing script: {viewing_script_path}")
    print(f"[INFO]")
    print(f"[INFO] To view all tracking results together:")
    print(f"[INFO]   bash {viewing_script_path}")
    
    return {
        "combined_rrd": combined_rrd_path,
        "viewing_script": viewing_script_path,
        "individual_rrds": rrd_paths,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine multiple RRD files")
    parser.add_argument(
        "rrd_files",
        type=Path,
        nargs="+",
        help="Input .rrd files to combine",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output combined .rrd file path",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        nargs="+",
        default=None,
        help="Entity path prefixes (e.g., 'mask_0/' 'mask_1/')",
    )
    parser.add_argument(
        "--create-script",
        action="store_true",
        help="Also create a viewing script",
    )
    
    args = parser.parse_args()
    
    # Combine RRD files
    combined_path = combine_rrd_files_python(
        input_rrd_paths=args.rrd_files,
        output_rrd_path=args.output,
        entity_prefixes=args.prefixes,
    )
    
    # Create viewing script if requested
    if args.create_script:
        script_path = args.output.parent / f"view_{args.output.stem}.sh"
        create_viewing_script(
            rrd_paths=args.rrd_files,
            output_script_path=script_path,
        )
    
    print(f"\n[INFO] Done!")
