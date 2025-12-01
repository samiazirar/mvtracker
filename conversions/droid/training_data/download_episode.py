import subprocess
import os
import json

# 1. PASTE YOUR METADATA HERE (As a Python Dictionary)
# I have included the entries from your snippet
episodes_data = {
    "AUTOLab+0d4edc83+2023-10-21-20h-00m-21s": {
        "relative_path": "AUTOLab/failure/2023-10-21/Sat_Oct_21_20:00:21_2023"
    },
    "AUTOLab+0d4edc83+2023-10-21-20h-16m-04s": {
        "relative_path": "AUTOLab/failure/2023-10-21/Sat_Oct_21_20:16:04_2023"
    },
    "AUTOLab+0d4edc83+2023-10-21-20h-16m-52s": {
        "relative_path": "AUTOLab/failure/2023-10-21/Sat_Oct_21_20:16:52_2023"
    },
    "AUTOLab+0d4edc83+2023-10-21-20h-18m-48s": {
        "relative_path": "AUTOLab/failure/2023-10-21/Sat_Oct_21_20:18:48_2023"
    }
}

# 2. CONFIGURATION
# Based on your note: ".../droid_raw/1.0.1/..."
BASE_BUCKET = "gs://gresearch/robotics/droid_raw/1.0.1" 
LOCAL_DOWNLOAD_DIR = "./droid_downloads"

def run_gsutil(command):
    """Helper to run shell commands"""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")

def download_episodes():
    for episode_id, data in episodes_data.items():
        rel_path = data['relative_path']
        
        # Construct the full Google Cloud Storage path
        # Removes trailing slashes to prevent double slash issues
        gcs_episode_path = f"{BASE_BUCKET}/{rel_path.strip('/')}"
        
        # Create a local folder for this episode to keep things organized
        local_episode_path = os.path.join(LOCAL_DOWNLOAD_DIR, rel_path)
        os.makedirs(local_episode_path, exist_ok=True)
        
        print(f"\n--- Processing: {rel_path} ---")

        # --- A. Download trajectory.h5 ---
        print("Downloading trajectory.h5...")
        cmd_traj = f"gsutil cp \"{gcs_episode_path}/trajectory.h5\" \"{local_episode_path}/\""
        run_gsutil(cmd_traj)

        # --- B. Download Metadata JSON (using wildcard) ---
        print("Downloading metadata json...")
        # Note: We quote the path to ensure the wildcard * is passed to gsutil, not expanded by local shell
        cmd_meta = f"gsutil cp \"{gcs_episode_path}/metadata*.json\" \"{local_episode_path}/\""
        run_gsutil(cmd_meta)

        # --- C. Download SVO Directory (Recursive) ---
        # The SVOs are usually in /recordings/SVO/ inside the episode
        print("Downloading SVO recordings...")
        
        # Create local SVO directory structure
        local_svo_path = os.path.join(local_episode_path, "recordings", "SVO")
        os.makedirs(local_svo_path, exist_ok=True)
        
        # Download the contents of the SVO folder
        cmd_svo = f"gsutil -m cp -r \"{gcs_episode_path}/recordings/SVO/*\" \"{local_svo_path}/\""
        run_gsutil(cmd_svo)

if __name__ == "__main__":
    download_episodes()
    print("\nBatch download complete.")