#!/usr/bin/env python3
"""
Modified from CUT3R: https://github.com/CUT3R/CUT3R

Online Human-Scene Reconstruction Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D scene point clouds and SMPLX sequences with the SceneHumanViewer. 
Use the command-line arguments to adjust parameters 
such as the model checkpoint path, image sequence directory, image size, device, etc.

Example:
    python demo.py --model_path src/human3r_896L.pth --size 512 \
        --seq_path examples/GoodMornin1.mp4 --subsample 1 --vis_threshold 2 \
        --downsample_factor 1 --use_ttt3r --reset_interval 100
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy

from types import SimpleNamespace

from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import roma

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--msk_threshold",
        type=float,
        default=0.1,
        help="Mask threshold. Ranging from 0 to 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output results.",
    )
    parser.add_argument(
        "--save_scene",
        action="store_true",
        help="Save a single scene.pkl with everything the viewer needs, so replay.py can launch the viewer later without re-running inference.",
    )
    parser.add_argument(
        "--eval_vsmpl",
        action="store_true",
        help="Run the VolumetricSMPL penetration metric: query each scene point against the "
             "predicted SMPL volumes and print per-frame inside counts. Also paints penetrating "
             "points red in the viewer.",
    )
    parser.add_argument(
        "--opt_selfpen_begin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run begin-only post-process optimization for SMPL-X self-intersection using VolumetricSMPL.",
    )
    parser.add_argument(
        "--selfpen_begin_steps",
        type=int,
        default=10,
        help="Adam steps for begin-only VolumetricSMPL self-penetration optimization.",
    )
    parser.add_argument(
        "--selfpen_begin_lr",
        type=float,
        default=1e-3,
        help="Learning rate for begin-only VolumetricSMPL self-penetration optimization.",
    )
    parser.add_argument(
        "--selfpen_opt_scope",
        choices=("begin", "all"),
        default="begin",
        help="Optimize only the first valid SMPL frame, or every frame for visual before/after comparison.",
    )
    parser.add_argument(
        "--compare_selfpen_begin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay the initial pre-optimization SMPL mesh with the post-processed mesh in the viewer.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Save smpl mesh projection.",
    )
    parser.add_argument(
        "--render_video",
        action="store_true",
        help="Save smpl mesh projection video.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to use. Default is None (use all images).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for input images. Default is 1 (use all images).",
    )
    parser.add_argument(
        "--reset_interval", 
        type=int, 
        default=10000000
        )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Use TTT3R.",
        default=False
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=10,
        help="Point cloud downsample factor for the viewer",
    )
    parser.add_argument(
        "--smpl_downsample",
        type=int,
        default=1,
        help="SMPL sequence downsample factor for the viewer",
    )
    parser.add_argument(
        "--camera_downsample",
        type=int,
        default=1,
        help="Camera motion downsample factor for the viewer",
    )
    parser.add_argument(
        "--mask_morph",
        type=int,
        default=10,
        help="Mask morphology for the viewer",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Frames processed per inference chunk. Bounds peak input/output RAM.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for input frame decoding. 0 keeps everything in main process.",
    )
    return parser.parse_args()


class FrameDataset(torch.utils.data.Dataset):
    """Lazily produces one view dict per frame for the demo's streaming pipeline.

    Mirrors the no-raymap branch of the previous prepare_input but drops the
    [1, 6, H, W] NaN ray_map placeholder, which is unused on the lighter
    inference path (forward_step / _recurrent_rollout never read it).
    """

    def __init__(self, img_paths, size, img_res=None, reset_interval=10000000):
        self.img_paths = list(img_paths)
        self.size = size
        self.img_res = img_res
        self.reset_interval = reset_interval
        self._K_mhmr = None
        if img_res is not None:
            from dust3r.utils.geometry import get_camera_parameters
            self._K_mhmr = get_camera_parameters(img_res, device="cpu")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        from src.dust3r.utils.image import load_images, pad_image
        img_dict = load_images([self.img_paths[i]], size=self.size, verbose=False)[0]
        view = {
            "img": img_dict["img"],
            "true_shape": torch.from_numpy(img_dict["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((i + 1) % self.reset_interval == 0).unsqueeze(0),
        }
        if self.img_res is not None:
            view["img_mhmr"] = pad_image(view["img"], self.img_res)
            view["K_mhmr"] = self._K_mhmr
        return view


def _strip_view_for_output(view):
    """The accumulated outputs["views"] list is only read for `img` (used to
    build colors) and `reset` (used to drop overlap-views). Keeping the rest
    in the accumulator wastes ~8 MB/frame at size=512."""
    return {
        "img": view["img"].detach().cpu(),
        "reset": view["reset"].detach().cpu(),
    }

class ChunkProcessor:
    """Streams output post-processing one chunk at a time.

    Carries minimal state across chunks (cumulative camera-pose base, overlap
    drop boundary flag, frame index, lazily-built SMPL layer). Per-frame
    artifacts go directly into the viewer payload at fp16/uint8 — saved
    files (--save) stay at fp32, byte-identical with the previous
    one-shot prepare_output.
    """

    def __init__(
        self,
        outdir,
        save=False,
        render=False,
        render_video=False,
        img_res=None,
        subsample=1,
        device='cuda',
        compute_vsmpl_metric=False,
        opt_selfpen_begin=True,
        selfpen_begin_steps=10,
        selfpen_begin_lr=1e-3,
        selfpen_opt_scope="begin",
        compare_selfpen_begin=True,
        use_pose=True,
    ):
        from src.dust3r.utils.camera import pose_encoding_to_camera
        from src.dust3r.post_process import (
            attach_pretrained_volume,
            estimate_focal_knowing_depth,
            optimize_smpl_selfpen,
        )
        from src.dust3r.utils.geometry import geotrf
        from src.dust3r.utils import SMPL_Layer, vis_heatmap, render_meshes
        from src.dust3r.utils.image import unpad_image
        from src.dust3r.utils.streaming import streaming_pose_recovery
        from viser_utils import get_color
        self._attach_pretrained_volume = attach_pretrained_volume
        self._pose_encoding_to_camera = pose_encoding_to_camera
        self._estimate_focal_knowing_depth = estimate_focal_knowing_depth
        self._optimize_smpl_selfpen = optimize_smpl_selfpen
        self._geotrf = geotrf
        self._SMPL_Layer = SMPL_Layer
        self._vis_heatmap = vis_heatmap
        self._render_meshes = render_meshes
        self._unpad_image = unpad_image
        self._streaming_pose_recovery = streaming_pose_recovery
        self._get_color = get_color

        self.outdir = outdir
        self.save = save
        self.render = render
        self.render_video = render_video
        self.img_res = img_res
        self.subsample = subsample
        self.device = device
        self.compute_vsmpl_metric = compute_vsmpl_metric
        self.opt_selfpen_begin = opt_selfpen_begin
        self.selfpen_begin_steps = selfpen_begin_steps
        self.selfpen_begin_lr = selfpen_begin_lr
        self.selfpen_opt_scope = selfpen_opt_scope
        self.compare_selfpen_begin = compare_selfpen_begin
        self.use_pose = use_pose

        # Cross-chunk state.
        self.pose_base = torch.eye(4)
        self.any_reset_seen = False
        self.prev_chunk_last_reset = False
        self.frame_counter = 0
        self.selfpen_begin_done = False

        # Lazily built on first chunk (need num_betas from a real prediction).
        self.smpl_layer = None
        self.smpl_faces = None
        self.vol_device = None

        # Viewer payload accumulators (compressed dtypes).
        self.viewer_pts3ds = []
        self.viewer_colors = []
        self.viewer_conf = []
        self.viewer_msks = []
        self.viewer_verts = []
        self.viewer_initial_verts = []
        self.viewer_smpl_id = []
        self.viewer_inside_masks = []
        # Per-chunk camera arrays, concatenated in finalize().
        self.cam_focal_chunks = []
        self.cam_pp_chunks = []
        self.cam_R_chunks = []
        self.cam_t_chunks = []

        self.H = None
        self.W = None

        if save:
            print(f"Saving output to {outdir}...")
            os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
            os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
            os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
            os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
            os.makedirs(os.path.join(outdir, "smpl"), exist_ok=True)
        if render:
            os.makedirs(os.path.join(outdir, "color_smpl"), exist_ok=True)

    def _ensure_smpl_layer(self, num_betas):
        if self.smpl_layer is not None:
            return
        self.smpl_layer = self._SMPL_Layer(
            type='smplx', gender='neutral', num_betas=num_betas,
            kid=False, person_center='head',
        )
        self.smpl_faces = self.smpl_layer.bm_x.faces
        if self.compute_vsmpl_metric or self.opt_selfpen_begin:
            self._attach_pretrained_volume(self.smpl_layer.bm_x, self.device)
            self.vol_device = next(self.smpl_layer.bm_x.volume.parameters()).device
            self.smpl_layer.to(self.vol_device)
        else:
            self.vol_device = "cpu"

    def _should_optimize_selfpen(self):
        if not self.opt_selfpen_begin or self.selfpen_begin_steps <= 0:
            return False
        return self.selfpen_opt_scope == "all" or not self.selfpen_begin_done

    def _optimize_selfpen(self, pose, shape, expression, f_id_global):
        if (
            not self._should_optimize_selfpen()
            or pose.shape[0] == 0
        ):
            return pose

        if self.selfpen_opt_scope == "begin":
            self.selfpen_begin_done = True
        opt_pose, stats = self._optimize_smpl_selfpen(
            self.smpl_layer.bm_x,
            pose,
            shape,
            expression,
            steps=self.selfpen_begin_steps,
            lr=self.selfpen_begin_lr,
        )
        print(
            f"[VolumetricSMPL:selfpen] Frame {f_id_global:06d}: "
            f"{stats['initial_loss']:.6f} -> {stats['final_loss']:.6f} "
            f"in {stats['steps']} steps on {stats['device']}"
        )
        return opt_pose

    def process_chunk(self, chunk_pred, chunk_views):
        """chunk_pred: list of res_cpu dicts (one per frame).
           chunk_views: list of stripped view dicts ({"img", "reset"})."""
        if not chunk_pred:
            return

        t_chunk_start = time.time()
        t_smpl_fwd = 0.0
        t_vsmpl = 0.0
        t_save = 0.0
        t_render = 0.0
        t_viewer = 0.0

        # Step A — drop overlap views at chunk boundary (replaces the
        # shifted_reset_mask trick of the old prepare_output).
        local_reset = torch.cat([v["reset"] for v in chunk_views], 0)  # [B_in]
        prev_last_for_next = bool(local_reset[-1].item())
        shifted = torch.cat(
            [torch.tensor([self.prev_chunk_last_reset]), local_reset[:-1]], dim=0
        )
        keep = ~shifted
        chunk_pred = [p for p, k in zip(chunk_pred, keep.tolist()) if k]
        chunk_views = [v for v, k in zip(chunk_views, keep.tolist()) if k]
        reset_mask = local_reset[keep]
        self.prev_chunk_last_reset = prev_last_for_next
        if not chunk_pred:
            return  # entire chunk was a single overlap-view

        B = len(chunk_pred)

        # Step B — extract tensors.
        pts3ds_self_ls = [p["pts3d_in_self_view"] for p in chunk_pred]
        pts3ds_other_ls = [p["pts3d_in_other_view"] for p in chunk_pred]
        conf_self_ls = [p["conf_self"] for p in chunk_pred]
        conf_other_ls = [p["conf"] for p in chunk_pred]
        pts3ds_self = torch.cat(pts3ds_self_ls, 0)  # [B, H, W, 3]
        if self.H is None:
            _, self.H, self.W, _ = pts3ds_self.shape
        H, W = self.H, self.W

        # Step C — pose recovery via shared streaming helper. Mirrors the old
        # prepare_output's two branches (raw poses if no reset seen anywhere
        # yet, streaming matmul once any reset fires) byte-for-byte.
        raw_poses = [
            self._pose_encoding_to_camera(p["camera_pose"].clone()).cpu()
            for p in chunk_pred
        ]  # list of [1, 4, 4]
        pr_poses, self.pose_base, self.any_reset_seen = self._streaming_pose_recovery(
            raw_poses, reset_mask, self.pose_base, self.any_reset_seen,
        )

        # Step D — pose-transform other-view points (use_pose path).
        if self.use_pose:
            transformed = []
            for pose, pself in zip(pr_poses, pts3ds_self):
                transformed.append(self._geotrf(pose, pself.unsqueeze(0)))
            pts3ds_other_ls = transformed
            conf_other_ls = conf_self_ls

        # Step E — focal estimation (per-frame along B; per-chunk == one-shot).
        pp = torch.tensor(
            [W // 2, H // 2], device=pts3ds_self.device
        ).float().repeat(B, 1)
        focal = self._estimate_focal_knowing_depth(
            pts3ds_self, pp, focal_mode="weiszfeld",
        )

        # Step F — build save-time tensors (fp32).
        depths_tosave = pts3ds_self[..., 2]                          # [B, H, W]
        pts3ds_other_tosave = torch.cat(pts3ds_other_ls)             # [B, H, W, 3]
        conf_self_tosave = torch.cat(conf_self_ls)                   # [B, H, W]
        conf_other_tosave = torch.cat(conf_other_ls)                 # [B, H, W]
        colors_tosave = torch.cat([
            0.5 * (v["img"].permute(0, 2, 3, 1) + 1.0) for v in chunk_views
        ])                                                           # [B, H, W, 3]
        cam2world_tosave = torch.cat(pr_poses)                       # [B, 4, 4]
        intrinsics_tosave = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
        intrinsics_tosave[:, 0, 0] = focal.detach()
        intrinsics_tosave[:, 1, 1] = focal.detach()
        intrinsics_tosave[:, 0, 2] = pp[:, 0]
        intrinsics_tosave[:, 1, 2] = pp[:, 1]

        # Step G — SMPL params.
        smpl_shape = [
            p.get("smpl_shape", torch.empty(1, 0, 10))[0] for p in chunk_pred
        ]
        smpl_rotvec = [
            roma.rotmat_to_rotvec(
                p.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0]
            )
            for p in chunk_pred
        ]
        smpl_transl = [
            p.get("smpl_transl", torch.empty(1, 0, 3))[0] for p in chunk_pred
        ]
        smpl_expression = [
            p.get("smpl_expression", [None])[0] for p in chunk_pred
        ]
        smpl_id = [p.get("smpl_id", torch.empty(1, 0))[0] for p in chunk_pred]

        if self.render or self.save:
            smpl_scores = [
                p.get("smpl_scores", torch.zeros(1, H, W, 1))[..., 0]
                for p in chunk_pred
            ]
            if self.img_res is not None:
                smpl_scores = [self._unpad_image(s, [H, W])[0] for s in smpl_scores]

        has_mask = "msk" in chunk_pred[0]
        if has_mask:
            msks = [p["msk"][..., 0] for p in chunk_pred]
            if self.img_res is not None:
                msks = [self._unpad_image(m, [H, W]) for m in msks]
        else:
            msks = [torch.zeros(1, H, W) for _ in range(B)]

        # Step H — lazy SMPL layer construction.
        self._ensure_smpl_layer(num_betas=smpl_shape[0].shape[-1])
        smpl_layer = self.smpl_layer
        smpl_faces = self.smpl_faces
        vol_device = self.vol_device

        if self.compute_vsmpl_metric or self.opt_selfpen_begin:
            intrinsics_dev = intrinsics_tosave.to(vol_device)
            pts3ds_other_dev = pts3ds_other_tosave.to(vol_device)
            pr_poses_dev = [p.to(vol_device) for p in pr_poses]
            msks_dev = [m.to(vol_device) for m in msks]
            smpl_rotvec_dev = [t.to(vol_device) for t in smpl_rotvec]
            smpl_shape_dev = [t.to(vol_device) for t in smpl_shape]
            smpl_transl_dev = [t.to(vol_device) for t in smpl_transl]
            smpl_expression_dev = [
                t.to(vol_device) if t is not None else None for t in smpl_expression
            ]
        else:
            intrinsics_dev = intrinsics_tosave
            smpl_rotvec_dev = smpl_rotvec
            smpl_shape_dev = smpl_shape
            smpl_transl_dev = smpl_transl
            smpl_expression_dev = smpl_expression

        # Step I — per-frame loop. Mirrors the old prepare_output body.
        for f_id_local in range(B):
            f_id_global = self.frame_counter + f_id_local
            n_humans_i = smpl_shape[f_id_local].shape[0]

            if n_humans_i > 0:
                initial_vert_world = None
                should_optimize_selfpen = self._should_optimize_selfpen()
                if self.compare_selfpen_begin and should_optimize_selfpen:
                    _t = time.time()
                    with torch.no_grad():
                        initial_smpl_out = smpl_layer(
                            smpl_rotvec_dev[f_id_local],
                            smpl_shape_dev[f_id_local],
                            smpl_transl_dev[f_id_local],
                            None, None,
                            K=intrinsics_dev[f_id_local].expand(n_humans_i, -1, -1),
                            expression=smpl_expression_dev[f_id_local],
                        )
                    t_smpl_fwd += time.time() - _t
                    initial_vert_world = self._geotrf(
                        pr_poses[f_id_local],
                        initial_smpl_out['smpl_v3d'].cpu().unsqueeze(0),
                    )[0]

                if should_optimize_selfpen:
                    smpl_rotvec_dev[f_id_local] = self._optimize_selfpen(
                        smpl_rotvec_dev[f_id_local],
                        smpl_shape_dev[f_id_local],
                        smpl_expression_dev[f_id_local],
                        f_id_global,
                    )
                    smpl_rotvec[f_id_local] = smpl_rotvec_dev[f_id_local].cpu()
                _t = time.time()
                with torch.no_grad():
                    smpl_out = smpl_layer(
                        smpl_rotvec_dev[f_id_local],
                        smpl_shape_dev[f_id_local],
                        smpl_transl_dev[f_id_local],
                        None, None,
                        K=intrinsics_dev[f_id_local].expand(n_humans_i, -1, -1),
                        expression=smpl_expression_dev[f_id_local],
                    )
                t_smpl_fwd += time.time() - _t

            depth = depths_tosave[f_id_local].numpy()
            conf = conf_self_tosave[f_id_local].numpy()
            color = colors_tosave[f_id_local].numpy()
            c2w = cam2world_tosave[f_id_local].numpy()
            intrins = intrinsics_tosave[f_id_local].numpy()

            inside_mask_np = None
            if n_humans_i > 0:
                smpl_v3d_cpu = smpl_out['smpl_v3d'].cpu()
                vert_world = self._geotrf(
                    pr_poses[f_id_local], smpl_v3d_cpu.unsqueeze(0),
                )[0]
                if initial_vert_world is None:
                    initial_vert_world = vert_world
                elif should_optimize_selfpen:
                    vert_delta = (vert_world - initial_vert_world).norm(dim=-1)
                    print(
                        f"[VolumetricSMPL:selfpen] Frame {f_id_global:06d}: "
                        f"vertex delta mean/max="
                        f"{vert_delta.mean().item() * 100:.3f}/"
                        f"{vert_delta.max().item() * 100:.3f} cm"
                    )
                pr_verts = [t.numpy() for t in smpl_v3d_cpu.unbind(0)]
                pr_faces = [smpl_faces] * n_humans_i

                if self.compute_vsmpl_metric:
                    _t = time.time()
                    verts_world_dev = self._geotrf(
                        pr_poses_dev[f_id_local], smpl_out['smpl_v3d'].unsqueeze(0),
                    )[0]
                    joints_world_dev = self._geotrf(
                        pr_poses_dev[f_id_local], smpl_out['smpl_j3d'].unsqueeze(0),
                    )[0]
                    points_world_dev = pts3ds_other_dev[f_id_local].reshape(-1, 3)
                    points_world_b = points_world_dev.unsqueeze(0).expand(
                        n_humans_i, -1, -1,
                    )
                    smpl_volume_state = SimpleNamespace(
                        full_pose=smpl_rotvec_dev[f_id_local].reshape(n_humans_i, -1),
                        joints=joints_world_dev,
                        vertices=verts_world_dev,
                    )
                    with torch.no_grad():
                        smpl_layer.bm_x.volume.detach_cache()
                        sdf = smpl_layer.bm_x.volume.query(
                            points_world_b, smpl_volume_state,
                        )
                    inside = (sdf < 0).any(dim=0)
                    human_mask = msks_dev[f_id_local].reshape(-1) > 0.5
                    scene_mask = ~human_mask
                    n_total = inside.numel()
                    n_inside_total = int(inside.sum().item())
                    n_inside_scene = int((inside & scene_mask).sum().item())
                    n_scene = int(scene_mask.sum().item())
                    print(
                        f"[VolumetricSMPL] Frame {f_id_global:06d}: "
                        f"scene-only {n_inside_scene}/{n_scene} inside, "
                        f"all {n_inside_total}/{n_total} inside, "
                        f"sdf min/med/max={sdf.min().item():.3f}/"
                        f"{sdf.median().item():.3f}/{sdf.max().item():.3f}"
                    )
                    inside_mask_np = inside.reshape(H, W).cpu().numpy()
                    t_vsmpl += time.time() - _t
            else:
                pr_verts = []
                pr_faces = []
                vert_world = torch.empty(0)
                initial_vert_world = torch.empty(0)
                if self.compute_vsmpl_metric:
                    inside_mask_np = np.zeros((H, W), dtype=bool)
                    print(f"[VolumetricSMPL] Frame {f_id_global:06d}: no humans detected")

            if self.render:
                _t = time.time()
                hm = self._vis_heatmap(
                    colors_tosave[f_id_local], smpl_scores[f_id_local],
                ).numpy()
                img_array_np = (color * 255).astype(np.uint8)
                smpl_rend = self._render_meshes(
                    img_array_np.copy(), pr_verts, pr_faces,
                    {'focal': intrins[[0, 1], [0, 1]],
                     'princpt': intrins[[0, 1], [-1, -1]]},
                    color=[self._get_color(i) / 255 for i in smpl_id[f_id_local]],
                )
                if has_mask:
                    msk_array_np = self._vis_heatmap(
                        colors_tosave[f_id_local], msks[f_id_local][0],
                    ).numpy()
                    color_smpl = np.concatenate([
                        img_array_np,
                        (msk_array_np * 255).astype(np.uint8),
                        (hm * 255).astype(np.uint8),
                        smpl_rend,
                    ], 1)
                else:
                    color_smpl = np.concatenate([
                        img_array_np,
                        (hm * 255).astype(np.uint8),
                        smpl_rend,
                    ], 1)
                t_render += time.time() - _t

            if self.save:
                _t = time.time()
                np.save(
                    os.path.join(self.outdir, "depth", f"{f_id_global:06d}.npy"),
                    depth,
                )
                np.save(
                    os.path.join(self.outdir, "conf", f"{f_id_global:06d}.npy"),
                    conf,
                )
                iio.imwrite(
                    os.path.join(self.outdir, "color", f"{f_id_global:06d}.png"),
                    (color * 255).astype(np.uint8),
                )
                np.savez(
                    os.path.join(self.outdir, "camera", f"{f_id_global:06d}.npz"),
                    pose=c2w, intrinsics=intrins,
                )
                np.savez(
                    os.path.join(self.outdir, "smpl", f"{f_id_global:06d}.npz"),
                    scores=smpl_scores[f_id_local].numpy(),
                    msk=msks[f_id_local].numpy() if has_mask else None,
                    shape=smpl_shape[f_id_local].numpy(),
                    rotvec=smpl_rotvec[f_id_local].numpy(),
                    transl=smpl_transl[f_id_local].numpy(),
                    expression=(
                        smpl_expression[f_id_local].numpy()
                        if smpl_expression[f_id_local] is not None else None
                    ),
                )
                t_save += time.time() - _t

            if self.render:
                _t = time.time()
                iio.imwrite(
                    os.path.join(self.outdir, "color_smpl", f"{f_id_global:06d}.png"),
                    color_smpl,
                )
                t_render += time.time() - _t

            # Append to the viewer payload in compressed dtypes. Shapes match
            # the old run_inference numpy outputs: pts3d/colors are
            # [1, H, W, 3], conf/msks are [1, H, W].
            _t = time.time()
            self.viewer_pts3ds.append(
                pts3ds_other_tosave[f_id_local:f_id_local + 1].numpy().astype(np.float16)
            )
            color_4d = colors_tosave[f_id_local:f_id_local + 1].numpy()
            self.viewer_colors.append(
                (color_4d.clip(0, 1) * 255 + 0.5).astype(np.uint8)
            )
            self.viewer_conf.append(
                conf_other_tosave[f_id_local:f_id_local + 1].numpy().astype(np.float16)
            )
            self.viewer_msks.append(msks[f_id_local].numpy().astype(np.float16))
            self.viewer_verts.append(
                vert_world.numpy() if isinstance(vert_world, torch.Tensor) else vert_world
            )
            self.viewer_initial_verts.append(
                initial_vert_world.numpy()
                if isinstance(initial_vert_world, torch.Tensor)
                else initial_vert_world
            )
            self.viewer_smpl_id.append(smpl_id[f_id_local].cpu().numpy())
            if self.compute_vsmpl_metric and inside_mask_np is not None:
                self.viewer_inside_masks.append(inside_mask_np)
            t_viewer += time.time() - _t

        # Per-chunk camera arrays.
        R_c2w = torch.cat([p[:, :3, :3] for p in pr_poses], 0)
        t_c2w = torch.cat([p[:, :3, 3] for p in pr_poses], 0)
        self.cam_focal_chunks.append(focal.cpu().numpy())
        self.cam_pp_chunks.append(pp.cpu().numpy())
        self.cam_R_chunks.append(R_c2w.cpu().numpy())
        self.cam_t_chunks.append(t_c2w.cpu().numpy())

        self.frame_counter += B

        t_total = time.time() - t_chunk_start
        t_other = max(
            0.0,
            t_total - (t_smpl_fwd + t_vsmpl + t_save + t_render + t_viewer),
        )
        print(
            f"  [proc] B={B} total={t_total:.2f}s | "
            f"smpl_fwd={t_smpl_fwd:.2f}s vsmpl={t_vsmpl:.2f}s "
            f"save={t_save:.2f}s render={t_render:.2f}s "
            f"viewer_append={t_viewer:.2f}s other={t_other:.2f}s"
        )

    def finalize(self, render_video=False, subsample=1):
        cam_dict = {
            "focal": np.concatenate(self.cam_focal_chunks, 0)
                if self.cam_focal_chunks else np.empty(0, dtype=np.float32),
            "pp": np.concatenate(self.cam_pp_chunks, 0)
                if self.cam_pp_chunks else np.empty((0, 2), dtype=np.float32),
            "R": np.concatenate(self.cam_R_chunks, 0)
                if self.cam_R_chunks else np.empty((0, 3, 3), dtype=np.float32),
            "t": np.concatenate(self.cam_t_chunks, 0)
                if self.cam_t_chunks else np.empty((0, 3), dtype=np.float32),
        }
        if self.render and render_video:
            print(f"Saving smpl mesh projection to {self.outdir}...")
            frames_dir = os.path.join(self.outdir, "color_smpl")
            video_path = os.path.join(self.outdir, "output_video.mp4")
            output_fps = 30 // subsample
            os.system(
                f'/usr/bin/ffmpeg -y -framerate {output_fps} -i "{frames_dir}/%06d.png" '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                f'-movflags +faststart -b:v 5000k "{video_path}"'
            )
        return (
            self.viewer_pts3ds,
            self.viewer_colors,
            self.viewer_conf,
            cam_dict,
            self.viewer_verts,
            self.viewer_initial_verts,
            self.smpl_faces,
            self.viewer_smpl_id,
            self.viewer_msks,
            self.viewer_inside_masks,
        )

def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    if device == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gib = props.total_memory / (1024 ** 3)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            f"[device] Using CUDA: {props.name} (cuda:{idx}, {total_gib:.1f} GiB), "
            f"CUDA_VISIBLE_DEVICES={cvd}, torch={torch.__version__}, "
            f"cuda_runtime={torch.version.cuda}"
        )
    else:
        print(f"[device] Using CPU (torch={torch.__version__})")

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model after adding the ckpt path. Streaming inference drives
    # model.forward_step directly, so we no longer need inference_recurrent_lighter.
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.streaming import frame_iter_from_loader
    from viser_utils import SceneHumanViewer

    timings = {}

    # Prepare image file paths.
    t0 = time.time()
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    timings["load_frames"] = time.time() - t0
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    t0 = time.time()
    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]
    timings["prepare_image_paths"] = time.time() - t0

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    t0 = time.time()
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()
    timings["load_model"] = time.time() - t0

    # Build streaming dataset + loader.
    print("Building frame dataset...")
    t0 = time.time()
    img_res = getattr(model, 'mhmr_img_res', None)
    dataset = FrameDataset(
        img_paths=img_paths,
        size=args.size,
        img_res=img_res,
        reset_interval=args.reset_interval,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=lambda x: x,
    )
    timings["build_dataset"] = time.time() - t0

    # Streaming inference + chunked post-processing. Each chunk is post-
    # processed (pose recovery, SMPL forward, optional save, viewer-payload
    # build at fp16/uint8) and dropped before the next chunk runs, so peak
    # RAM is bounded by chunk_size rather than total frame count.
    # no_grad + autocast(enabled=False) match the wrapping that
    # inference_recurrent_lighter used to provide (inference.py).
    print("Running inference + processing (streaming)...")
    processor = ChunkProcessor(
        outdir=args.output_dir,
        save=args.save,
        render=args.render,
        render_video=args.render_video,
        img_res=img_res,
        subsample=args.subsample,
        device=device,
        compute_vsmpl_metric=args.eval_vsmpl,
        opt_selfpen_begin=args.opt_selfpen_begin,
        selfpen_begin_steps=args.selfpen_begin_steps,
        selfpen_begin_lr=args.selfpen_begin_lr,
        selfpen_opt_scope=args.selfpen_opt_scope,
        compare_selfpen_begin=args.compare_selfpen_begin,
        use_pose=True,
    )
    start_time = time.time()
    state_args = None
    trackers = None
    chunk_pred = []
    chunk_views = []
    n_processed = 0
    n_chunks = 0
    chunk_inf_time = 0.0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
        for view in frame_iter_from_loader(loader):
            t_inf = time.time()
            res, state_args, trackers = model.forward_step(
                view, device,
                state_args=state_args, trackers=trackers,
                use_ttt3r=args.use_ttt3r,
            )
            chunk_inf_time += time.time() - t_inf
            chunk_pred.append(res)
            chunk_views.append(_strip_view_for_output(view))
            n_processed += 1
            if len(chunk_pred) >= args.chunk_size:
                t_proc = time.time()
                processor.process_chunk(chunk_pred, chunk_views)
                t_proc = time.time() - t_proc
                n_chunks += 1
                print(
                    f"  chunk {n_chunks} ({n_processed} frames done): "
                    f"inference={chunk_inf_time:.2f}s | processing={t_proc:.2f}s"
                )
                chunk_inf_time = 0.0
                chunk_pred.clear()
                chunk_views.clear()
        if chunk_pred:
            t_proc = time.time()
            processor.process_chunk(chunk_pred, chunk_views)
            t_proc = time.time() - t_proc
            n_chunks += 1
            print(
                f"  chunk {n_chunks} (tail, {len(chunk_pred)} frames): "
                f"inference={chunk_inf_time:.2f}s | processing={t_proc:.2f}s"
            )
            chunk_pred.clear()
            chunk_views.clear()
    total_time = time.time() - start_time
    timings["run_inference_and_process"] = total_time
    per_frame_time = total_time / max(n_processed, 1)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Collect viewer payload from the processor. Arrays are already numpy in
    # compressed dtypes (pts3d/conf/msks: fp16; colors: uint8).
    (pts3ds_to_vis,
     colors_to_vis,
     conf_to_vis,
     cam_dict,
     verts_to_vis,
     initial_verts_to_vis,
     smpl_faces,
     smpl_id,
     msks_to_vis,
     inside_masks) = processor.finalize(
        render_video=args.render_video, subsample=args.subsample,
    )
    edge_colors = [None] * len(pts3ds_to_vis)

    if args.eval_vsmpl:
        # Paint points inside an SMPL volume in red. colors_to_vis[i] is uint8
        # [1, H, W, 3]; inside_masks[i] is numpy bool [H, W].
        red = np.array([255, 0, 0], dtype=np.uint8)
        for i, m in enumerate(inside_masks):
            colors_to_vis[i][0][m] = red

    if args.save_scene:
        import pickle
        os.makedirs(args.output_dir, exist_ok=True)
        scene_path = os.path.join(args.output_dir, "scene.pkl")
        scene_bundle = {
            "pts3ds": pts3ds_to_vis,
            "colors": colors_to_vis,
            "conf": conf_to_vis,
            "cam_dict": cam_dict,
            "verts": verts_to_vis,
            "initial_verts": initial_verts_to_vis,
            "smpl_faces": smpl_faces,
            "smpl_id": smpl_id,  # already numpy from processor.finalize
            "msks": msks_to_vis,
            "size": args.size,
        }
        with open(scene_path, "wb") as f:
            pickle.dump(scene_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Scene bundle saved to {scene_path}")

    # Create and run the point cloud viewer.
    print("Launching Human3R viewer...")
    t0 = time.time()
    viewer = SceneHumanViewer(
        pts3ds_to_vis,
        colors_to_vis,
        conf_to_vis,
        cam_dict,
        verts_to_vis,
        smpl_faces,
        smpl_id,
        msks_to_vis,
        gt_smpl_verts=(
            initial_verts_to_vis
            if args.compare_selfpen_begin and args.opt_selfpen_begin
            else None
        ),
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        show_gt_smpl=args.compare_selfpen_begin and args.opt_selfpen_begin,
        gt_smpl_label="Initial SMPL",
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample,
    )
    timings["create_viewer_and_run"] = time.time() - t0

    print("\n===== Runtime report =====")
    for name, t in timings.items():
        print(f"  {name:25s} {t:8.2f} s")
    print("==========================")

    viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)

if __name__ == "__main__":
    main()
