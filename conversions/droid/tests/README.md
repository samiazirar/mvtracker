# Training Data Export Tests

This folder contains optional, data-dependent tests for the training data exporters and video generation.

## Requirements
- Access to a DROID episode (H5 + SVOs + cam2base JSON).
- GPU available for depth export and video generation.
- Environment variables:
  - `DROID_TEST_CONFIG` (required): path to a config YAML pointing at a single episode.
  - `DROID_TEST_EXPORT_RGB_DEPTH=1` (optional): enable the RGB/depth PNG export tests (uses ZED depth).
  - `DROID_TEST_FULL_SEQUENCE=1` (optional): run full-sequence exports (can be slow/heavy).
  - `DROID_TEST_VIDEO=1` (optional): run `create_video_with_tracks.py` to produce a short video (defaults to 5 frames; override with `DROID_TEST_VIDEO_MAX_FRAMES`).

## What the tests do
- `export_tracks` (1-frame): copies the config, caps `max_frames` to 1, writes outputs to a temp directory, runs the tracks exporter, and checks that `tracks.npz`, `extrinsics.npz`, and `quality.json` exist. Records runtime.
- `export_tracks` (full, optional): runs with full sequence (`max_frames=0`) when `DROID_TEST_FULL_SEQUENCE=1` is set; verifies full length and records runtime.
- `export_rgb_depth_pngs` (1-frame, optional): copies the config with `max_frames=1`, runs the PNG exporter, and checks that at least one RGB and depth PNG are written. Records runtime. Requires `DROID_TEST_EXPORT_RGB_DEPTH=1`.
- `export_rgb_depth_pngs` (full, optional): full sequence when both `DROID_TEST_EXPORT_RGB_DEPTH=1` and `DROID_TEST_FULL_SEQUENCE=1` are set; verifies outputs and records runtime.
- `create_video_with_tracks` (optional): runs a short video generation (default 5 frames) when `DROID_TEST_VIDEO=1` is set; checks that a video file is produced and records runtime. Override frame cap with `DROID_TEST_VIDEO_MAX_FRAMES`.

## Running
From repo root:
```bash
export DROID_TEST_CONFIG=/path/to/config.yaml
pytest conversions/droid/tests -q
```
Optional envs enable additional tests:
- `export DROID_TEST_EXPORT_RGB_DEPTH=1` for PNG export
- `export DROID_TEST_FULL_SEQUENCE=1` for full-length runs
- `export DROID_TEST_VIDEO=1` for video generation (optionally `DROID_TEST_VIDEO_MAX_FRAMES=5`)
