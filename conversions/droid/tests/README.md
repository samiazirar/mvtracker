# Training Data Export Tests

This folder contains optional, data-dependent tests for the training data exporters and video generation.

## Requirements
- Access to a DROID episode (H5 + SVOs + cam2base JSON). Default config points to example paths; override if needed.
- GPU available for depth export and video generation.
- Optional environment overrides:
  - `DROID_TEST_CONFIG`: path to a config YAML (defaults to `conversions/droid/training_data/example_config.yaml`).
  - `DROID_TEST_VIDEO_MAX_FRAMES`: frame cap for video/end-to-end tests (default 3 for end-to-end, 5 for standalone video).

## What the tests do
- `export_tracks` (1-frame): copies the config, caps `max_frames` to 1, writes outputs to a temp directory, runs the tracks exporter, and checks that `tracks.npz`, `extrinsics.npz`, and `quality.json` exist. Records runtime.
- `export_tracks` (full): runs with full sequence (`max_frames=0`), verifies full length, records runtime.
- `export_rgb_depth_pngs` (1-frame): copies the config with `max_frames=1`, runs the PNG exporter, and checks that at least one RGB and depth PNG are written. Records runtime.
- `export_rgb_depth_pngs` (full): full sequence, verifies outputs, records runtime.
- `create_video_with_tracks`: runs a short video generation (default 5 frames; override with `DROID_TEST_VIDEO_MAX_FRAMES`), checks that a video file is produced, records runtime.
- `end-to-end RRD/video`: runs and writes outputs under `point_clouds/tests`, ensuring RRD, tracks NPZ, and video are produced.

## Running
From repo root:
```bash
export DROID_TEST_CONFIG=/path/to/config.yaml
pytest conversions/droid/tests -q
```
Optional envs:
- `export DROID_TEST_CONFIG=/path/to/config.yaml` to override default
- `export DROID_TEST_VIDEO_MAX_FRAMES=5` to adjust video frame caps
