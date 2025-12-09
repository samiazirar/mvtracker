from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
import os

repo_id = "Yeshenglong/InternSpatial"

# 1. Find the name of the first parquet file dynamically
print("Finding the first file name...")
files = list_repo_files(repo_id=repo_id, repo_type="dataset")
first_parquet_file = next(f for f in files if f.endswith(".parquet"))
print(f"Target file found: {first_parquet_file}")

# 2. Download that specific file to /tmp
print(f"Downloading {first_parquet_file} to /tmp (this may take a moment)...")
local_path = hf_hub_download(
    repo_id=repo_id,
    filename=first_parquet_file,
    repo_type="dataset",
    local_dir="/tmp",
    local_dir_use_symlinks=False
)

print(f"File saved to: {local_path}")

# 3. Open the LOCAL file and read only the first batch
# We use standard python open(), not fs.open()
with open(local_path, "rb") as f:
    parquet_file = pq.ParquetFile(f)
    
    # Grab only the first 10 rows without loading the whole 2GB file into RAM
    first_batch = next(parquet_file.iter_batches(batch_size=10))
    
    df = first_batch.to_pandas()

print("\nFirst 10 rows loaded successfully:")
print(df)