# TCP Position Computation: Before vs After

## BEFORE (FK-based approach)
```
┌─────────────────────────────────────────────────────────────┐
│  1. Get joint angles from scene.get_joint_angles_aligned()  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Update robot model with joint angles                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Compute Forward Kinematics (FK)                          │
│     - Uses URDF model (may have simplifications)             │
│     - Propagates joint angles through kinematic chain        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Extract TCP transform from FK result                     │
│     - May have accumulated errors                            │
│     - Depends on URDF accuracy                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Pass TCP to _compute_gripper_bbox()                      │
└─────────────────────────────────────────────────────────────┘

⚠️  Issues:
   - URDF model may be simplified/approximate
   - FK computation can accumulate errors
   - Not using the official high-fidelity robot data
```

## AFTER (API-based approach)
```
┌─────────────────────────────────────────────────────────────┐
│  1. Call scene.get_tcp_aligned(timestamp)                    │
│     ✓ Direct from robot controller recordings                │
│     ✓ Pre-processed by dataset maintainers                   │
│     ✓ High-fidelity sensor data                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Convert 7D pose to 4x4 matrix                            │
│     - [x, y, z, qx, qy, qz, qw] → 4x4 transform             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Pass TCP to _compute_gripper_bbox()                      │
└─────────────────────────────────────────────────────────────┘

✓  Benefits:
   - Uses official robot controller data
   - No URDF model approximations
   - More accurate position and orientation
   - Faster (fewer computation steps)
   - "Ground truth" TCP position
```

## Key Difference

### The TCP (Tool Center Point) is the exact operational center between gripper fingers

**OLD:** Computed indirectly via mathematical model (FK + URDF)
- Subject to model inaccuracies
- Accumulated kinematic errors
- Generic URDF may not match actual robot

**NEW:** Retrieved directly from official dataset API
- Recorded from actual robot controller
- Pre-processed and aligned by dataset creators
- Represents true gripper position during recording

## Result
The gripper bounding boxes now:
- ✓ Show the correct bottom contact surface position
- ✓ Have accurate orientation matching real gripper
- ✓ Are properly positioned in all dimensions (x, y, z)
- ✓ Align with where gripper actually makes contact with objects
