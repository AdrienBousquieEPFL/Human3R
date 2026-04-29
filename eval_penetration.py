#!/usr/bin/env python3
"""
Depth-based human-scene inter-penetration metric

For each frame, we ask: are SMPL vertices farther from the camera than the
scene surface at the same pixel? If yes, that vertex is sitting behind the
visible surface = inside a wall / floor / object. We report the mean
penetration depth in millimetres.

Usage:
    python eval_penetration.py --scene /path/to/outputs/.../scene.pkl
    python eval_penetration.py --scene scene.pkl --tolerance_mm 5 --save_csv pen.csv
"""

import argparse
import pickle

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Path to scene.pkl from demo.py --save_scene")
    p.add_argument("--tolerance_mm", type=float, default=5.0,
                   help="Penetration depth (mm) below which we ignore (surface noise). Default 5mm.")
    p.add_argument("--no_mask", action="store_true",
                   help="Don't filter human pixels out of the scene depth. "
                        "Beware: back-of-body verts will be flagged as penetrating front-of-body.")
    p.add_argument("--msk_threshold", type=float, default=0.1,
                   help="Pixels with msk > this are treated as human (matches demo.py viewer default).")
    p.add_argument("--conf_threshold", type=float, default=0.0,
                   help="Drop scene pixels with confidence below this.")
    p.add_argument("--save_csv", type=str, default=None,
                   help="Optional CSV path for per-frame results.")
    p.add_argument("--unit_scale_to_m", type=float, default=1.0,
                   help="Multiplier from scene units to metres. Default 1.0 (Human3R is approximately metric).")
    return p.parse_args()


def world_to_camera(points_world, R_c2w, t_c2w):
    """Transform (..., 3) world points to camera coords using a c2w pose."""
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    flat = points_world.reshape(-1, 3)
    cam = flat @ R_w2c.T + t_w2c
    return cam.reshape(points_world.shape)


def compute_frame_penetration(
    pts3d, verts, focal, pp, R_c2w, t_c2w,
    msk=None, conf=None,
    msk_threshold=0.1, conf_threshold=0.0,
    tolerance_m=0.005, unit_scale_to_m=1.0,
):
    """
    Returns a dict of metrics for one frame, or None if no humans.

    pts3d:  (1, H, W, 3) world-space scene points (organised as a pixel grid)
    verts:  (N_humans, V, 3) or (V_total, 3) world-space SMPL vertices
    focal:  scalar pinhole focal length (in pixels)
    pp:     (2,) principal point [px, py]
    R_c2w:  (3, 3) cam-to-world rotation
    t_c2w:  (3,)   cam-to-world translation
    msk:    (1, H, W) human mask (high values = human) or None
    conf:   (1, H, W) confidence or None
    """
    if verts.size == 0:
        return None

    pts3d = pts3d[0]                              # (H, W, 3)
    H, W, _ = pts3d.shape

    # --- Step 1: scene depth image -----------------------------------------
    pts_cam = world_to_camera(pts3d, R_c2w, t_c2w)   # (H, W, 3)
    scene_depth = pts_cam[..., 2]                    # (H, W)

    # --- Step 2: validity mask for scene pixels ----------------------------
    valid_scene = np.isfinite(scene_depth) & (scene_depth > 0)
    if msk is not None:
        valid_scene &= (msk[0] <= msk_threshold)     # drop human pixels
    if conf is not None and conf_threshold > 0:
        valid_scene &= (conf[0] >= conf_threshold)

    # --- Step 3: project SMPL verts into the camera ------------------------
    verts_flat = verts.reshape(-1, 3)
    n_verts = verts_flat.shape[0]
    v_cam = world_to_camera(verts_flat, R_c2w, t_c2w)   # (V_total, 3)
    z = v_cam[:, 2]
    in_front = z > 1e-6
    safe_z = np.where(in_front, z, 1.0)
    u = focal * v_cam[:, 0] / safe_z + pp[0]
    v = focal * v_cam[:, 1] / safe_z + pp[1]
    u_i = np.round(u).astype(np.int64)
    v_i = np.round(v).astype(np.int64)
    in_bounds = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    queryable = in_front & in_bounds

    # --- Step 4: depth comparison -----------------------------------------
    pen_depth = np.zeros(n_verts, dtype=np.float64)
    has_scene = np.zeros(n_verts, dtype=bool)
    if queryable.any():
        u_q = u_i[queryable]
        v_q = v_i[queryable]
        scene_d = scene_depth[v_q, u_q]
        scene_ok = valid_scene[v_q, u_q]
        pen_depth[queryable] = z[queryable] - scene_d
        has_scene[queryable] = scene_ok

    is_pen = has_scene & (pen_depth > tolerance_m)

    n_visible = int(has_scene.sum())
    n_pen = int(is_pen.sum())

    if n_pen > 0:
        pen_mm = pen_depth[is_pen] * unit_scale_to_m * 1000.0
        mean_mm = float(pen_mm.mean())
        max_mm = float(pen_mm.max())
    else:
        mean_mm = 0.0
        max_mm = 0.0

    return {
        "n_verts_total": int(n_verts),
        "n_visible": n_visible,
        "n_penetrating": n_pen,
        "pct_penetrating": (n_pen / n_visible) if n_visible > 0 else 0.0,
        "mean_penetration_mm": mean_mm,
        "max_penetration_mm": max_mm,
    }


def evaluate_sequence(
    pts3ds, verts_list, cam_dict, msks=None, conf_list=None,
    tolerance_mm=5.0, msk_threshold=0.1, conf_threshold=0.0,
    use_mask=True, unit_scale_to_m=1.0,
):
    """
    Compute the penetration metric over a full sequence.

    Inputs match the in-memory arrays from demo.py (after the .cpu().numpy() step):
        pts3ds:     list of (1, H, W, 3) world-space scene points
        verts_list: list of (N_humans, V, 3) world-space SMPL verts
        cam_dict:   {"focal": (B,), "pp": (B, 2), "R": (B, 3, 3), "t": (B, 3)}
        msks:       list of (1, H, W) human masks (high = human), or None
        conf_list:  list of (1, H, W) confidences, or None

    Returns:
        per_frame: list of per-frame metric dicts (None where no humans)
        summary:   aggregate dict
    """
    focals = np.asarray(cam_dict["focal"]).reshape(-1)
    pps = np.asarray(cam_dict["pp"])
    Rs = np.asarray(cam_dict["R"])
    ts = np.asarray(cam_dict["t"])
    n_frames = len(pts3ds)
    tol_m = tolerance_mm / 1000.0 / unit_scale_to_m

    per_frame = []
    for f_id in range(n_frames):
        result = compute_frame_penetration(
            pts3d=np.asarray(pts3ds[f_id]),
            verts=np.asarray(verts_list[f_id]),
            focal=float(focals[f_id]),
            pp=pps[f_id],
            R_c2w=Rs[f_id],
            t_c2w=ts[f_id],
            msk=np.asarray(msks[f_id]) if (msks is not None and use_mask) else None,
            conf=np.asarray(conf_list[f_id]) if conf_list is not None else None,
            msk_threshold=msk_threshold,
            conf_threshold=conf_threshold,
            tolerance_m=tol_m,
            unit_scale_to_m=unit_scale_to_m,
        )
        per_frame.append(result)

    valid = [r for r in per_frame if r is not None]
    if not valid:
        summary = {
            "n_frames_evaluated": 0,
            "n_frames_total": n_frames,
            "tolerance_mm": tolerance_mm,
            "use_mask": use_mask,
        }
    else:
        total_pen_sum = sum(r["mean_penetration_mm"] * r["n_penetrating"] for r in valid)
        total_n_pen = sum(r["n_penetrating"] for r in valid)
        total_visible = sum(r["n_visible"] for r in valid)
        summary = {
            "n_frames_evaluated": len(valid),
            "n_frames_total": n_frames,
            "tolerance_mm": tolerance_mm,
            "use_mask": use_mask,
            "mean_penetration_mm": total_pen_sum / max(total_n_pen, 1),
            "pct_visible_verts_penetrating": total_n_pen / max(total_visible, 1),
            "max_penetration_mm": max(r["max_penetration_mm"] for r in valid),
            "total_penetrating_verts": total_n_pen,
        }
    return per_frame, summary


def print_summary(summary):
    """Pretty-print a summary dict from evaluate_sequence."""
    print()
    print("=== Human-Scene Penetration (A1: body -> scene depth-buffer) ===")
    if summary.get("n_frames_evaluated", 0) == 0:
        print(f"  No frames with humans found ({summary['n_frames_total']} frames).")
        return
    print(f"  Frames evaluated:           {summary['n_frames_evaluated']} / {summary['n_frames_total']}")
    print(f"  Tolerance:                  {summary['tolerance_mm']:.1f} mm")
    print(f"  Mask humans from scene:     {summary['use_mask']}")
    print(f"  Mean penetration depth:     {summary['mean_penetration_mm']:.2f} mm  (over penetrating verts)")
    print(f"  % visible verts penetrate:  {summary['pct_visible_verts_penetrating'] * 100:.2f}%")
    print(f"  Max single-vertex pen:      {summary['max_penetration_mm']:.2f} mm")
    print(f"  Total penetrating verts:    {summary['total_penetrating_verts']}")


def save_per_frame_csv(per_frame, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame", "n_verts", "n_visible", "n_penetrating",
            "pct_penetrating", "mean_penetration_mm", "max_penetration_mm",
        ])
        for i, r in enumerate(per_frame):
            if r is None:
                w.writerow([i, 0, 0, 0, 0.0, 0.0, 0.0])
            else:
                w.writerow([
                    i, r["n_verts_total"], r["n_visible"], r["n_penetrating"],
                    r["pct_penetrating"], r["mean_penetration_mm"], r["max_penetration_mm"],
                ])


def main():
    args = parse_args()

    print(f"Loading scene from {args.scene}...")
    with open(args.scene, "rb") as f:
        s = pickle.load(f)

    per_frame, summary = evaluate_sequence(
        pts3ds=s["pts3ds"],
        verts_list=s["verts"],
        cam_dict=s["cam_dict"],
        msks=s["msks"],
        conf_list=s["conf"],
        tolerance_mm=args.tolerance_mm,
        msk_threshold=args.msk_threshold,
        conf_threshold=args.conf_threshold,
        use_mask=not args.no_mask,
        unit_scale_to_m=args.unit_scale_to_m,
    )
    print_summary(summary)

    if args.save_csv:
        save_per_frame_csv(per_frame, args.save_csv)
        print(f"  Per-frame CSV:              {args.save_csv}")


if __name__ == "__main__":
    main()
