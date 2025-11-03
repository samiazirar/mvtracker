#!/bin/bash
# Batch convert multiple BEHAVE scenes to .npz format

# Configuration
BEHAVE_ROOT="/data/behave-dataset/behave_all"
OUTPUT_DIR="./conversions/behave_converted"
MAX_FRAMES=100  # Limit frames for faster conversion, set to empty for all frames
DOWNSCALE_FACTOR=2
MASK_TYPE="person"

# List of scenes to convert (add more as needed)
SCENES=(
    "Date01_Sub01_backpack_back"
    "Date01_Sub01_backpack_hand"
    "Date01_Sub01_basketball"
    "Date01_Sub01_chairblack_sit"
)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Convert each scene
for scene in "${SCENES[@]}"; do
    echo "======================================"
    echo "Converting scene: $scene"
    echo "======================================"
    
    cmd="python conversions/behave_to_npz.py \
        --behave_root $BEHAVE_ROOT \
        --scene $scene \
        --output_dir $OUTPUT_DIR \
        --mask_type $MASK_TYPE \
        --downscale_factor $DOWNSCALE_FACTOR"
    
    if [ -n "$MAX_FRAMES" ]; then
        cmd="$cmd --max_frames $MAX_FRAMES"
    fi
    
    echo "Running: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully converted: $scene"
        
        # Optionally inspect the converted file
        echo "Inspecting converted file..."
        python conversions/inspect_behave_npz.py "$OUTPUT_DIR/$scene.npz"
    else
        echo "✗ Failed to convert: $scene"
    fi
    
    echo ""
done

echo "======================================"
echo "Batch conversion complete!"
echo "======================================"
echo "Converted files in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.npz
