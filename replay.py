#!/usr/bin/env python3
"""
Replay a saved Human3R scene in the viewer without re-running inference.

Run demo.py with --save_scene first to produce scene.pkl. Then:

    python replay.py --scene /path/to/outputs/20260424_120000/scene.pkl
"""

import argparse
import os
import pickle
import sys


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from viser_utils import SceneHumanViewer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a saved scene bundle and launch the Human3R viewer."
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Path to scene.pkl produced by demo.py --save_scene.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--msk_threshold", type=float, default=0.1)
    parser.add_argument("--mask_morph", type=int, default=10)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--smpl_downsample", type=int, default=1)
    parser.add_argument("--camera_downsample", type=int, default=1)
    parser.add_argument(
        "--compare_initial_smpl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the saved initial pre-optimization SMPL mesh when available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading scene bundle from {args.scene}...")
    with open(args.scene, "rb") as f:
        s = pickle.load(f)

    edge_colors = [None] * len(s["pts3ds"])
    initial_verts = s.get("initial_verts") if args.compare_initial_smpl else None

    print("Launching Human3R viewer...")
    viewer = SceneHumanViewer(
        s["pts3ds"],
        s["colors"],
        s["conf"],
        s["cam_dict"],
        s["verts"],
        s["smpl_faces"],
        s["smpl_id"],
        s["msks"],
        gt_smpl_verts=initial_verts,
        device=args.device,
        port=args.port,
        edge_color_list=edge_colors,
        show_camera=True,
        show_gt_smpl=initial_verts is not None,
        gt_smpl_label="Initial SMPL",
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=s.get("size", 512),
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample,
    )
    viewer.run()


if __name__ == "__main__":
    main()
