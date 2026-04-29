#!/usr/bin/env python3
"""
Replay a saved Human3R scene in the viewer without re-running inference.

Run demo.py with --save_scene first to produce scene.pkl. Then:

    python replay.py --scene /path/to/outputs/20260424_120000/scene.pkl
"""

import argparse
import pickle

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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading scene bundle from {args.scene}...")
    with open(args.scene, "rb") as f:
        s = pickle.load(f)

    edge_colors = [None] * len(s["pts3ds"])

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
        device=args.device,
        port=args.port,
        edge_color_list=edge_colors,
        show_camera=True,
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
