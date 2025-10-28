# Mask Lifting Functions Reorganization

## Summary

All mask lifting and visualization functions have been moved to a dedicated module `utils/mask_lifting_utils.py` for better code organization.

## Changes Made

### 1. New File: `utils/mask_lifting_utils.py`

Created a dedicated module containing:
- `lift_mask_to_3d()` - Lift single 2D mask to 3D point cloud
- `lift_mask_to_3d_batch()` - Batch version for multiple masks
- `visualize_mask_3d()` - Visualize mask in Rerun
- `visualize_masks_batch()` - Batch visualization for multiple masks

### 2. Updated: `utils/pointcloud_utils.py`

- **Removed**: `lift_mask_to_3d()` and `lift_mask_to_3d_batch()` function definitions (245 lines)
- **Added**: Import and re-export from `mask_lifting_utils` for backward compatibility
- **Updated**: Module docstring to note imported functions

### 3. Updated: `utils/visualization_utils.py`

- **Removed**: `visualize_mask_3d()` and `visualize_masks_batch()` function definitions (206 lines)
- **Added**: Import and re-export from `mask_lifting_utils` for backward compatibility
- **Updated**: Module docstring to note imported functions

### 4. Updated Scripts

All scripts updated to import from the new module:
- `demo_mask_lifting.py`
- `lift_and_visualize_masks.py`
- `visualize_hand_masks.py`

## Usage

### Primary Import (Recommended)
```python
from utils.mask_lifting_utils import (
    lift_mask_to_3d,
    lift_mask_to_3d_batch,
    visualize_mask_3d,
    visualize_masks_batch,
)
```

### Backward Compatible Imports
```python
# Still works for existing code
from utils.pointcloud_utils import lift_mask_to_3d, lift_mask_to_3d_batch
from utils.visualization_utils import visualize_mask_3d, visualize_masks_batch
```

## Benefits

1. **Better Organization**: Mask lifting functions are now in their own dedicated module
2. **Cleaner Code**: Reduced duplication and clarified module responsibilities
3. **Backward Compatibility**: Existing code continues to work through re-exports
4. **Easier Maintenance**: All mask-related functionality in one place

## Verification

All changes verified with:
- ✅ Syntax checks passed
- ✅ Import tests passed
- ✅ Re-export functionality verified
- ✅ All dependent scripts tested

## Module Responsibilities

- **`utils/mask_lifting_utils.py`**: Primary location for mask lifting and visualization
- **`utils/pointcloud_utils.py`**: 3D point cloud processing (imports mask lifting for convenience)
- **`utils/visualization_utils.py`**: Rerun visualization and video export (imports mask viz for convenience)
