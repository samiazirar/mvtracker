import rerun as rr
import numpy as np
import argparse
import sys
import pandas as pd
from scipy.spatial import cKDTree

def get_points_from_rrd(rrd_path, entity_path, timeline="frame_index"):
    """
    Extracts point clouds from an .rrd file using the Dataframe API.
    Returns a dictionary: { frame_index: (N, 3) numpy array }
    """
    print(f"[INFO] Loading {rrd_path}...")
    try:
        # Load the recording
        recording = rr.dataframe.load_recording(rrd_path)
        
        # Create a view for the specific entity
        # 'contents' is strictly required as a keyword argument in newer Rerun versions
        view = recording.view(
            index=timeline,
            contents=entity_path
        )
        
        # Read all data into a pandas dataframe
        # The columns will typically be named like '{entity_path}:Points3D:positions'
        df = view.select().read_pandas()
        
        # Find the column containing the 3D positions
        # Look for either ':Position3D' or ':Points3D:positions'
        pos_col = [c for c in df.columns if "Position3D" in c or ":positions" in c]
        if not pos_col:
            print(f"[WARN] No position data found for {entity_path} in {rrd_path}")
            print(f"       Available columns: {df.columns.tolist()}")
            return {}
        pos_col = pos_col[0]

        # Extract data
        points_by_frame = {}
        for index, row in df.iterrows():
            # The dataframe index is the timeline (frame_index)
            frame = int(index)
            
            # Positions are stored as arrays/lists in the cell
            raw_points = row[pos_col]
            
            if isinstance(raw_points, (np.ndarray, list)) and len(raw_points) > 0:
                # Flatten/reshape if necessary depending on how pyarrow returns it
                points = np.vstack(raw_points)
                points_by_frame[frame] = points
                
        return points_by_frame
        
    except Exception as e:
        print(f"[ERROR] Failed to read {rrd_path}: {e}")
        return {}

def compute_chamfer_distance(pts_a, pts_b):
    """
    Computes the symmetric Chamfer distance between two point clouds.
    Lower is better (0 = identical).
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return None

    # Nearest neighbor from A to B
    tree_b = cKDTree(pts_b)
    dist_a_to_b, _ = tree_b.query(pts_a)
    
    # Nearest neighbor from B to A
    tree_a = cKDTree(pts_a)
    dist_b_to_a, _ = tree_a.query(pts_b)
    
    # Mean of both directions
    return np.mean(dist_a_to_b) + np.mean(dist_b_to_a)

def main():
    parser = argparse.ArgumentParser(description="Compare two Rerun .rrd files.")
    parser.add_argument("old_rrd", help="Path to the original (old) .rrd file")
    parser.add_argument("new_rrd", help="Path to the new .rrd file")
    parser.add_argument("--entity", default="world/wrist_cam/points", help="Entity path to compare")
    parser.add_argument("--output", default="comparison_result.rrd", help="Output RRD file")
    args = parser.parse_args()

    # FIX 1: Disable spawn to prevent crash in Docker/Headless environment
    rr.init("rrd_comparison", spawn=False)
    
    # 2. Extract Data
    data_old = get_points_from_rrd(args.old_rrd, args.entity)
    data_new = get_points_from_rrd(args.new_rrd, args.entity)
    
    if not data_old or not data_new:
        print("[ERROR] Failed to load point cloud data from one or both files.")
        return

    common_frames = sorted(list(set(data_old.keys()) & set(data_new.keys())))
    print(f"[INFO] Found {len(common_frames)} common frames.")

    distances = []

    # 3. Re-log and Compare
    for i in common_frames:
        rr.set_time_sequence("frame_index", i)
        
        pts_old = data_old[i]
        pts_new = data_new[i]

        # -- Visual Comparison --
        # Log Old as RED
        rr.log(
            "comparison/old_version", 
            rr.Points3D(pts_old, colors=[255, 0, 0], radii=0.005),
        )
        
        # Log New as GREEN
        rr.log(
            "comparison/new_version", 
            rr.Points3D(pts_new, colors=[0, 255, 0], radii=0.005),
        )

        # -- Quantitative Comparison --
        dist = compute_chamfer_distance(pts_old, pts_new)
        if dist is not None:
            distances.append(dist)
            # Log the error as a scalar curve
            rr.log("metrics/chamfer_distance", rr.Scalar(dist))

    if distances:
        print(f"=== Alignment Results ===")
        print(f"Mean Chamfer Distance: {np.mean(distances):.4f} m")
        print(f"(Lower is closer. If ~0.0, they are identical. If large, the fix worked/changed things.)")
        
    # FIX 2: Save to file explicitly
    rr.save(args.output)
    print(f"[SUCCESS] Result saved to: {args.output}")

if __name__ == "__main__":
    main()