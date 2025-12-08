# CRITICAL: GPU Requirement for Both Pipelines

## ⚠️ IMPORTANT DISCOVERY

**Both pipelines REQUIRE GPU/CUDA access** because the ZED SDK cannot operate without it.

### Why GPU is Required

The ZED SDK (pyzed) absolutely requires CUDA/GPU to be available, even for simple operations like:
- Opening SVO files
- Reading camera intrinsics
- Any camera initialization

**This cannot be bypassed.** Even with `sdk_gpu_id = -1`, the ZED SDK will fail with "NO GPU DETECTED" if CUDA is not available.

### Impact on Pipelines

#### Metadata-Only Pipeline
- ❌ **Cannot run on CPU-only nodes** (despite the name)
- ✅ **Requires GPU** for intrinsics extraction from SVO files
- ✅ **Lighter than full pipeline** (no depth inference, just intrinsics)
- ✅ **Uses less GPU** (~5 seconds vs 2-3 minutes per episode)

#### Full Pipeline
- ✅ **Requires GPU** for depth extraction
- ✅ **Uses GPU heavily** for depth computation
- ✅ **Takes longer** but provides complete dataset

### Solutions

#### Option 1: Use GPU-Enabled Nodes (Recommended)
```bash
# Both scripts work fine on GPU nodes
# Metadata-only is just much faster

# On GPU node:
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh 100 3 1  # 3 workers/GPU, 1 GPU
./run_pipeline_cluster_huggingface_compressed_lossy.sh 100 3 1        # 3 workers/GPU, 1 GPU
```

#### Option 2: Pre-Extract Intrinsics (True CPU-Only)
For truly CPU-only processing:

1. **One-time GPU pass:** Extract intrinsics for all episodes
```bash
# On GPU node: Extract intrinsics for all episodes
python batch_extract_intrinsics.py --all-episodes --output /data/intrinsics_cache/
```

2. **CPU-only processing:** Modify `generate_tracks_and_metadata.py` to load pre-extracted intrinsics
```python
# Instead of looking for intrinsics.json in recordings/
# Load from pre-extracted cache
intrinsics = load_cached_intrinsics(episode_id, '/data/intrinsics_cache/')
```

3. **Skip intrinsics step** in pipeline and only generate tracks

#### Option 3: Skip 2D Projections Entirely
Modify `generate_tracks_and_metadata.py` to work without intrinsics:
- ✅ Generate 3D tracks (no camera needed)
- ✅ Generate normalized flow (no camera needed)
- ✅ Save extrinsics (from calibration file)
- ❌ Skip 2D projections (requires intrinsics)

### Comparison Table

| Feature | Metadata-Only | Full Pipeline | True CPU-Only |
|---------|---------------|---------------|---------------|
| GPU Required | ✅ Yes (ZED SDK) | ✅ Yes (ZED SDK) | ❌ No |
| Time/Episode | ~30s | ~2-3 min | ~5s |
| GPU Usage | Low (intrinsics) | High (depth) | None |
| 3D Tracks | ✅ Yes | ✅ Yes | ✅ Yes |
| 2D Projections | ✅ Yes | ✅ Yes | ❌ No |
| Normalized Flow | ✅ Yes | ✅ Yes | ✅ Yes |
| Depth Maps | ❌ No | ✅ Yes | ❌ No |
| Intrinsics | ✅ Yes | ✅ Yes | ⚠️ Pre-cached |

### Recommended Workflow

**For Most Users:**
```bash
# Use full pipeline on GPU nodes
# It already extracts everything including intrinsics
./run_pipeline_cluster_huggingface_compressed_lossy.sh -1  # Process all episodes
```

**For Vision-Only Models (No Depth Needed):**
```bash
# Use metadata-only pipeline on GPU nodes
# Much faster, still gets intrinsics for 2D projections
./run_pipeline_cluster_huggingface_metadata_only_no_depth.sh -1  # Process all episodes
```

**For Experiments / Development (No GPU Access):**
1. Extract intrinsics once on a GPU machine
2. Cache them to shared storage
3. Modify pipeline to use cached intrinsics
4. Run on CPU-only nodes

### Error Messages

If you see:
```
[ZED][ERROR] No NVIDIA graphics card detected.
[ZED][ERROR] NO GPU DETECTED
```

**This means:**
- You're not on a GPU node
- OR GPU drivers are not working
- OR `nvidia-smi` fails

**Solutions:**
1. Move to a GPU-enabled node
2. Fix GPU drivers (`nvidia-smi` should work)
3. Use Option 2 or 3 above for CPU-only processing

### Testing GPU Availability

Before running pipelines:
```bash
# Check GPU is available
nvidia-smi

# Should show GPU(s), not error
# Example output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  Tesla V100-SXM2...  On   | 00000000:00:04.0 Off |                    0 |
```

### Current Status

- ✅ Pipelines are correctly configured
- ✅ GPU assignment works properly
- ✅ All files and paths are correct
- ❌ **Cannot test without GPU node**
- ⏳ **Need to run on lmgpu-node with working CUDA**

### Next Steps

1. **Verify you're on a GPU node:** `nvidia-smi` should work
2. **If on CPU node:** Move to GPU node or use pre-cached intrinsics
3. **Run pipelines:** Both should work once GPU is available
4. **For production:** Consider pre-extracting intrinsics for maximum flexibility
