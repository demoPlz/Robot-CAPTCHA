#!/usr/bin/env python3
"""Set an object's pose in a USD file.

Usage (run in the any6d conda env):
    conda run --no-capture-output -n any6d python scripts/set_usd_object_pose.py \
        --usd public/assets/usd/sorting.usd \
        --prim /World/cyan_container \
        --pos 0.2603 -0.2102 0.0410 \
        --quat-xyzw 0.0035 -0.0015 0.7091 0.7051
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation as R


def main():
    parser = argparse.ArgumentParser(description="Set object pose in a USD file")
    parser.add_argument("--usd", required=True, help="Path to USD file")
    parser.add_argument("--prim", required=True, help="Prim path (e.g. /World/cyan_container)")
    parser.add_argument("--pos", nargs=3, type=float, required=True, help="World position x y z (meters)")
    parser.add_argument("--quat-xyzw", nargs=4, type=float, required=True, help="Quaternion x y z w")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a .bak backup")
    args = parser.parse_args()

    usd_path = Path(args.usd).resolve()
    if not usd_path.exists():
        print(f"❌ USD file not found: {usd_path}")
        return

    # Backup
    if not args.no_backup:
        bak = usd_path.with_suffix(usd_path.suffix + ".bak")
        shutil.copy2(usd_path, bak)
        print(f"📦 Backup saved to {bak}")

    target_pos = np.array(args.pos)
    target_quat_xyzw = np.array(args.quat_xyzw)  # [x, y, z, w]

    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(args.prim)
    if not prim or not prim.IsValid():
        print(f"❌ Prim not found: {args.prim}")
        return

    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()

    print(f"📋 Current xform ops for {args.prim}:")
    for op in ops:
        print(f"   {op.GetOpName()} = {op.Get()}")

    # Find the relevant ops
    translate_op = None
    orient_op = None
    pivot_op = None
    scale_units_op = None

    for op in ops:
        name = op.GetOpName()
        if name == "xformOp:translate" and not op.IsInverseOp():
            translate_op = op
        elif name == "xformOp:orient":
            orient_op = op
        elif name == "xformOp:translate:pivot":
            pivot_op = op
        elif name == "xformOp:scale:unitsResolve":
            scale_units_op = op

    if translate_op is None:
        print("❌ No xformOp:translate found")
        return

    # Get pivot and units scale
    pivot = np.array([50.0, 50.0, 50.0])  # default
    units_scale = 0.001  # default

    if pivot_op is not None:
        pv = pivot_op.Get()
        pivot = np.array([pv[0], pv[1], pv[2]])
        print(f"   Pivot: {pivot}")

    if scale_units_op is not None:
        sv = scale_units_op.Get()
        units_scale = sv[0]  # assume uniform
        print(f"   Units scale: {units_scale}")

    # The xform chain for point p_local through this prim:
    #   p_world = translate + pivot + Orient * Scale * unitsScale * (p_local - pivot)
    #
    # For the prim origin (p_local = 0):
    #   p_world = translate + pivot + Orient * (unitsScale * (-pivot))
    #   p_world = translate + pivot + Orient(-unitsScale * pivot)
    #
    # So:
    #   translate = target_pos - pivot - Orient(-unitsScale * pivot)

    rot = R.from_quat(target_quat_xyzw)  # scipy uses [x,y,z,w]
    rotated_term = rot.apply(-units_scale * pivot)

    new_translate = target_pos - pivot - rotated_term
    print(f"\n🔧 Computing new translate:")
    print(f"   target_pos      = {target_pos}")
    print(f"   pivot            = {pivot}")
    print(f"   rot(-scale*piv) = {rotated_term}")
    print(f"   new_translate   = {new_translate}")

    # Verify
    verify_pos = new_translate + pivot + rotated_term
    print(f"   ✓ verification  = {verify_pos} (should match target_pos)")

    # Set translate
    translate_op.Set(Gf.Vec3d(*new_translate.tolist()))

    # Set orient (USD uses w, x, y, z)
    if orient_op is not None:
        w, x, y, z = target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]
        orient_op.Set(Gf.Quatf(float(w), float(x), float(y), float(z)))
        print(f"   Set orient (w,x,y,z) = ({w:.4f}, {x:.4f}, {y:.4f}, {z:.4f})")
    else:
        print("   ⚠️  No orient op found, rotation not set")

    # Save
    stage.GetRootLayer().Save()
    print(f"\n✅ Saved {usd_path}")

    # Readback verification
    stage2 = Usd.Stage.Open(str(usd_path))
    prim2 = stage2.GetPrimAtPath(args.prim)
    xfc = UsdGeom.XformCache()
    mat = xfc.GetLocalToWorldTransform(prim2)
    print(f"   Readback LocalToWorld: {mat}")


if __name__ == "__main__":
    main()
