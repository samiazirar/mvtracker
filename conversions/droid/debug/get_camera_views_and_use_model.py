"""
Compares two Rerun .rrd files by:
1. Automatically discovering all entities containing Point Clouds (Position3D).
2. Matching entities between the Old and New files.
3. Visualizing them side-by-side (Red vs Green).
4. Calculating Chamfer Distance for each camera/entity to quantify alignment.
"""

import rerun as rr
import numpy as np
import argparse
import pandas as pd
from scipy.spatial import cKDTree

def load_recording_and_find_entities(rrd_path, timeline="frame_index"):
    """
    Loads an RRD file and identifies all entities that contain 3D positions.
    Returns:
        recording: The Rerun recording object.
        entities: A list of entity paths (strings) that contain Position3D data.
    """
    print(f"[INFO] Loading {rrd_path}...")
    recording = rr.dataframe.load_recording(rrd_path)
    
    # We create a view of the whole recording to inspect the schema (columns)
    # We only care about the specific timeline
    try:
        view = recording.view(index=timeline)
        # We read just one row or the schema to find columns, 
        # but to be safe regarding sparse data, we can just inspect column names 
        # from a light query or the recording schema if accessible. 
        # Robust way: Read everything (if files aren't massive) or use the schema.
        # For simplicity/robustness in this script:
        df_schema = view.select().read_pandas()
    except Exception as e:
        print(f"[ERROR] Could not read recording {rrd_path}: {e}")
        return recording, []

    # Find columns that indicate 3D points
    # Column format is usually: 'entity_path:ComponentType'
    # e.g., 'world/points/wrist_cam:Position3D'
    point_entities = set()
    for col in df_schema.columns:
        if "Position3D" in col:
            # Extract the entity path (everything before the last colon)
            entity_path = col.rsplit(':', 1)[0]
            point_entities.add(entity_path)
    
    print(f"      -> Found {len(point_entities)} point cloud entities.")
    return recording, sorted(list(point_entities))

def extract_points(recording, entity_path, timeline="frame_index"):
    """
    Extracts point clouds for a specific entity from a loaded recording.
    Returns: { frame_index: (N, 3) numpy array }
    """
    view = recording.view(
        index=timeline,
        contents=entity_path
    )
    
    df = view.select().read_pandas()
    
    # Find the specific Position3D column for this entity
    pos_col = [c for c in df.columns if entity_path in c and "Position3D" in c]
    if not pos_col:
        return {}
    pos_col = pos_col[0]

    points_by_frame = {}
    for index, row in df.iterrows():
        raw_points = row[pos_col]
        
        # Check if valid data exists for this frame
        if isinstance(raw_points, (np.ndarray, list)) and len(raw_points) > 0:
            # Rerun dataframe often returns a flat array or list of arrays depending on version
            points = np.vstack(raw_points)
            points_by_frame[int(index)] = points
            
    return points_by_frame

def compute_chamfer_distance(pts_a, pts_b):
    """
    Computes Symmetric Chamfer Distance.
    
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return None

    # A -> B
    tree_b = cKDTree(pts_b)
    dist_a_to_b, _ = tree_b.query(pts_a)
    
    # B -> A
    tree_a = cKDTree(pts_a)
    dist_b_to_a, _ = tree_a.query(pts_b)
    
    return np.mean(dist_a_to_b) + np.mean(dist_b_to_a)

def main():
    parser = argparse.ArgumentParser(description="Compare alignment of ALL cameras between two RRD files.")
    parser.add_argument("old_rrd", help="Path to the original (old) .rrd file")
    parser.add_argument("new_rrd", help="Path to the new .rrd file")
    args = parser.parse_args()

    # 1. Initialize Rerun Viewer
    rr.init("alignment_checker", spawn=True)
    
    # 2. Load Recordings and Discover Entities
    rec_old, entities_old = load_recording_and_find_entities(args.old_rrd)
    rec_new, entities_new = load_recording_and_find_entities(args.new_rrd)

    # 3. Find Common Entities
    common_entities = sorted(list(set(entities_old) & set(entities_new)))
    
    if not common_entities:
        print("[ERROR] No common point cloud entities found between the two files!")
        print(f"Old: {entities_old}")
        print(f"New: {entities_new}")
        return

    print(f"\n[INFO] Comparing {len(common_entities)} entities: {common_entities}\n")

    summary_stats = []

    # 4. Loop over every common camera/entity
    for entity in common_entities:
        print(f"Processing: {entity} ...")
        
        # Extract data
        data_old = extract_points(rec_old, entity)
        data_new = extract_points(rec_new, entity)
        
        common_frames = sorted(list(set(data_old.keys()) & set(data_new.keys())))
        
        if not common_frames:
            print(f"  [WARN] No common frames for {entity}. Skipping.")
            continue

        entity_distances = []

        # Process Frames
        for i in common_frames:
            rr.set_time_sequence("frame_index", i)
            
            pts_old = data_old[i]
            pts_new = data_new[i]

            # A. Visual Comparison
            # We strip 'world/' to make the visualization hierarchy cleaner
            vis_name = entity.replace("world/", "")
            
            # Log Old (Red)
            rr.log(
                f"comparison/{vis_name}/old_red", 
                rr.Points3D(pts_old, colors=[255, 0, 0], radii=0.005),
            )
            
            # Log New (Green)
            rr.log(
                f"comparison/{vis_name}/new_green", 
                rr.Points3D(pts_new, colors=[0, 255, 0], radii=0.005),
            )

            # B. Metric Calculation
            dist = compute_chamfer_distance(pts_old, pts_new)
            if dist is not None:
                entity_distances.append(dist)
                # Log Metric
                rr.log(f"metrics/{vis_name}/chamfer_dist", rr.Scalar(dist))

        # Stats for this entity
        if entity_distances:
            mean_dist = np.mean(entity_distances)
            summary_stats.append({
                "Entity": entity,
                "Mean Chamfer (m)": mean_dist,
                "Frames": len(common_frames)
            })

    # 5. Print Summary
    print("\n" + "="*60)
    print(f"{'ALIGNMENT SUMMARY':^60}")
    print("="*60)
    print(f"{'Entity Name':<40} | {'Mean Error (m)':<15}")
    print("-" * 60)
    
    total_error = 0
    for stat in summary_stats:
        print(f"{stat['Entity']:<40} | {stat['Mean Chamfer (m)']:<15.5f}")
        total_error += stat['Mean Chamfer (m)']
    
    print("-" * 60)
    if summary_stats:
        print(f"{'AVERAGE ACROSS ALL CAMS':<40} | {total_error/len(summary_stats):<15.5f}")
    print("="*60)
    print("[INFO] Done. Check Rerun viewer for visual overlap.")

if __name__ == "__main__":
    main()