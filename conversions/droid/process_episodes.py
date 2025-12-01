import json
import argparse
from collections import defaultdict
import os

def process_episodes(json_path, data_root, limit):
    """
    Reads a DROID superset JSON, sorts episodes by quality per lab, 
    and prints their directory paths.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        print("Please provide the correct path to 'cam2base_extrinsic_superset.json'")
        return

    lab_episodes = defaultdict(list)
    for lab, episodes in data.items():
        for episode_path, details in episodes.items():
            episode_dir = os.path.join(data_root, lab, episode_path)
            quality = details.get('quality', 0)
            lab_episodes[lab].append((quality, episode_dir))

    sorted_episodes = []
    for lab in sorted(lab_episodes.keys()):
        # Sort by quality, descending
        episodes_for_lab = sorted(lab_episodes[lab], key=lambda x: x[0], reverse=True)
        if limit != -1:
            episodes_for_lab = episodes_for_lab[:limit]
        
        # We only need the path
        sorted_episodes.extend([e[1] for e in episodes_for_lab])
        
    for episode_dir in sorted_episodes:
        print(episode_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort DROID episodes by quality and print their paths.")
    parser.add_argument("--json_path", default="/data/droid/calib_and_annot/droid/cam2base_extrinsic_superset.json", help="Path to the superset JSON file.")
    parser.add_argument("--data_root", default="/data/droid", help="Root directory of the DROID data.")
    parser.add_argument("--limit", type=int, default=-1, help="Number of episodes per lab to process. -1 for all.")
    args = parser.parse_args()
    
    process_episodes(args.json_path, args.data_root, args.limit)
