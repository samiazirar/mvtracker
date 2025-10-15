#!/bin/bash
# Verification script to prove cfg3 vs cfg4 URDF difference is the root cause

echo "=========================================="
echo "Config 3 vs Config 4 URDF Analysis"
echo "=========================================="
echo ""

echo "1. Checking Config 3 URDF (ur5.urdf) for gripper links..."
echo "   grep -i 'finger\|pad\|knuckle\|wsg\|gripper' RH20T/models/ur5/urdf/ur5.urdf"
echo "   Result:"
if grep -i "finger\|pad\|knuckle\|wsg\|gripper" RH20T/models/ur5/urdf/ur5.urdf 2>/dev/null; then
    echo "   ✗ UNEXPECTED: Found gripper links in cfg3 URDF"
else
    echo "   ✓ CONFIRMED: NO gripper links found (exit code $?)"
fi
echo ""

echo "2. Checking Config 4 URDF (ur5_robotiq_85.urdf) for gripper links..."
echo "   grep -i 'finger\|pad' RH20T/models/ur5/urdf/ur5_robotiq_85.urdf | wc -l"
GRIPPER_LINKS=$(grep -i "finger\|pad" RH20T/models/ur5/urdf/ur5_robotiq_85.urdf 2>/dev/null | wc -l)
echo "   Result: $GRIPPER_LINKS lines containing finger/pad keywords"
if [ "$GRIPPER_LINKS" -gt 10 ]; then
    echo "   ✓ CONFIRMED: Config 4 has gripper links"
else
    echo "   ✗ UNEXPECTED: Config 4 should have many gripper links"
fi
echo ""

echo "3. Comparing total link counts..."
CFG3_LINKS=$(grep "<link name=" RH20T/models/ur5/urdf/ur5.urdf 2>/dev/null | wc -l)
CFG4_LINKS=$(grep "<link name=" RH20T/models/ur5/urdf/ur5_robotiq_85.urdf 2>/dev/null | wc -l)
echo "   Config 3 (ur5.urdf):          $CFG3_LINKS links"
echo "   Config 4 (ur5_robotiq_85.urdf): $CFG4_LINKS links"
DIFF=$((CFG4_LINKS - CFG3_LINKS))
echo "   Difference: $DIFF links"
if [ "$DIFF" -gt 5 ]; then
    echo "   ✓ CONFIRMED: Config 4 has $DIFF more links (gripper links)"
else
    echo "   ✗ UNEXPECTED: Should have significant difference"
fi
echo ""

echo "4. Checking specific finger pad links (required for bbox)..."
echo "   Looking for 'left_inner_finger_pad' in Config 3..."
if grep -q "left_inner_finger_pad" RH20T/models/ur5/urdf/ur5.urdf 2>/dev/null; then
    echo "   ✗ UNEXPECTED: Found in cfg3"
else
    echo "   ✓ CONFIRMED: NOT found in Config 3"
fi

echo "   Looking for 'left_inner_finger_pad' in Config 4..."
if grep -q "left_inner_finger_pad" RH20T/models/ur5/urdf/ur5_robotiq_85.urdf 2>/dev/null; then
    echo "   ✓ CONFIRMED: Found in Config 4"
else
    echo "   ✗ UNEXPECTED: Should be in cfg4"
fi

echo "   Looking for 'right_inner_finger_pad' in Config 3..."
if grep -q "right_inner_finger_pad" RH20T/models/ur5/urdf/ur5.urdf 2>/dev/null; then
    echo "   ✗ UNEXPECTED: Found in cfg3"
else
    echo "   ✓ CONFIRMED: NOT found in Config 3"
fi

echo "   Looking for 'right_inner_finger_pad' in Config 4..."
if grep -q "right_inner_finger_pad" RH20T/models/ur5/urdf/ur5_robotiq_85.urdf 2>/dev/null; then
    echo "   ✓ CONFIRMED: Found in Config 4"
else
    echo "   ✗ UNEXPECTED: Should be in cfg4"
fi
echo ""

echo "5. Comparing joint sequences from configs.json..."
echo "   Config 3 joint sequence:"
python3 -c "import json; data = json.load(open('RH20T/configs/configs.json')); cfg3 = [x for x in data if x['conf_num']==3][0]; print('   ', cfg3['robot_joint_sequence']); print('   ', f'Total: {len(cfg3[\"robot_joint_sequence\"])} joints')"

echo "   Config 4 joint sequence:"
python3 -c "import json; data = json.load(open('RH20T/configs/configs.json')); cfg4 = [x for x in data if x['conf_num']==4][0]; print('   ', cfg4['robot_joint_sequence']); print('   ', f'Total: {len(cfg4[\"robot_joint_sequence\"])} joints')"
echo ""

echo "=========================================="
echo "CONCLUSION"
echo "=========================================="
echo "Config 3 (WSG-50) URDF has NO gripper finger links."
echo "Config 4 (Robotiq 2F-85) URDF HAS gripper finger links."
echo ""
echo "The bbox computation function (_compute_gripper_bbox) requires"
echo "finger pad links (left_inner_finger_pad, right_inner_finger_pad)"
echo "to compute gripper orientation and position."
echo ""
echo "Without these links, Forward Kinematics cannot provide the"
echo "transforms needed, causing bbox computation to fail with:"
echo "  '[Warning] No gripper pad transforms available; cannot compute gripper bbox.'"
echo ""
echo "This is the DEFINITIVE root cause."
echo "=========================================="
