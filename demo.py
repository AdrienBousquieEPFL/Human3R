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

from VolumetricSMPL import attach_volume
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


def _frame_iter_from_loader(loader):
    """Yields each loaded view, then yields a deepcopy overlap-view (reset=False)
    immediately after any view whose reset==True. Reproduces the overlap-view
    side effect of the old prepare_input loop."""
    for view in loader:
        yield view
        if bool(view["reset"].item()):
            overlap = deepcopy(view)
            overlap["reset"] = torch.tensor(False).unsqueeze(0)
            yield overlap


def _strip_view_for_output(view):
    """The accumulated outputs["views"] list is only read for `img` (used to
    build colors) and `reset` (used to drop overlap-views). Keeping the rest
    in the accumulator wastes ~8 MB/frame at size=512."""
    return {
        "img": view["img"].detach().cpu(),
        "reset": view["reset"].detach().cpu(),
    }

def prepare_output(
        outputs, outdir, revisit=1, use_pose=True,
        save=False, render=False, render_video=False, img_res=None, subsample=1,
        device='cuda', compute_vsmpl_metric=False):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        save (bool): Whether to save output results.
        render (bool): Whether to save smpl mesh projection.
        render_video (bool): Whether to save smpl mesh projection video.
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.utils import SMPL_Layer, vis_heatmap, render_meshes
    from src.dust3r.utils.image import unpad_image
    from viser_utils import get_color

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"] for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"] for output in outputs["pred"]]
    conf_self = [output["conf_self"] for output in outputs["pred"]]
    conf_other = [output["conf"] for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    # reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # keeps only reset_mask=False pr_poses
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.numpy(),
        "pp": pp.numpy(),
        "R": R_c2w.numpy(),
        "t": t_c2w.numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach()
    intrinsics_tosave[:, 1, 1] = focal.detach()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # get SMPL parameters from outputs
    smpl_shape = [output.get(
        "smpl_shape", torch.empty(1,0,10))[0] for output in outputs["pred"]]
    smpl_rotvec = [roma.rotmat_to_rotvec(
        output.get(
            "smpl_rotmat", torch.empty(1,0,53,3,3))[0]) for output in outputs["pred"]]
    smpl_transl = [output.get(
        "smpl_transl", torch.empty(1,0,3))[0] for output in outputs["pred"]]
    smpl_expression = [output.get(
        "smpl_expression", [None])[0] for output in outputs["pred"]]
    smpl_id = [output.get(
        "smpl_id", torch.empty(1,0))[0] for output in outputs["pred"]]
    # smpl_loc = [output.get(
    #     "smpl_loc", torch.empty(1,0,2))[0] for output in outputs["pred"]]
    # K_mhmr = [output.get(
    #     "K_mhmr", torch.empty(1,0,3))[0] for output in outputs["views"]]
        
    if render or save:
        smpl_scores = [
            output.get("smpl_scores", torch.zeros(1, H, W, 1))[...,0] for output in outputs["pred"]]
        if img_res is not None:
            smpl_scores = [
                unpad_image(s, [H, W])[0] for s in smpl_scores]

    has_mask = "msk" in outputs["pred"][0]
    if has_mask:
        msks = [output["msk"][...,0] for output in outputs["pred"]]
        if img_res is not None:
            msks = [unpad_image(m, [H, W]) for m in msks]
    else:
        msks = [torch.zeros(1, H, W) for _ in range(B)]

    # SMPL layer
    smpl_layer = SMPL_Layer(type='smplx',
                            gender='neutral',
                            num_betas=smpl_shape[0].shape[-1],
                            kid=False,
                            person_center='head')
    smpl_faces = smpl_layer.bm_x.faces

    if compute_vsmpl_metric:
        attach_volume(smpl_layer.bm_x, pretrained=True, device=device)
        vol_device = next(smpl_layer.bm_x.volume.parameters()).device
        smpl_layer.to(vol_device)
        # Pre-move every tensor consumed by the per-frame volume query so we don't pay a
        # host->device transfer on each iteration. CPU originals are still referenced for
        # numpy / viewer code below.
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
        # No volumetric query — keep everything on CPU and reuse the originals for the
        # SMPL forward call. pr_poses_dev / pts3ds_other_dev / msks_dev are unused in this branch.
        intrinsics_dev = intrinsics_tosave
        smpl_rotvec_dev = smpl_rotvec
        smpl_shape_dev = smpl_shape
        smpl_transl_dev = smpl_transl
        smpl_expression_dev = smpl_expression

    if save:
        print(f"Saving output to {outdir}...")
        os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "smpl"), exist_ok=True)

    all_verts = []
    inside_masks = []  # per-frame [H, W] bool: scene points (non-human) inside any SMPL volume
    for f_id in range(B):
        n_humans_i = smpl_shape[f_id].shape[0]
        
        if n_humans_i > 0:
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvec_dev[f_id],
                    smpl_shape_dev[f_id],
                    smpl_transl_dev[f_id],
                    None, None,
                    K=intrinsics_dev[f_id].expand(n_humans_i, -1, -1),
                    expression=smpl_expression_dev[f_id])
        
        depth = depths_tosave[f_id].numpy()
        conf = conf_self_tosave[f_id].numpy()
        color = colors_tosave[f_id].numpy()
        c2w = cam2world_tosave[f_id].numpy()
        intrins = intrinsics_tosave[f_id].numpy()

        if n_humans_i > 0:
            # CPU copies for downstream (numpy / viewer / geotrf with CPU pr_poses)
            smpl_v3d_cpu = smpl_out['smpl_v3d'].cpu()
            # transform smpl verts to world coordinates
            all_verts.append(geotrf(pr_poses[f_id], smpl_v3d_cpu.unsqueeze(0))[0])
            pr_verts = [t.numpy() for t in smpl_v3d_cpu.unbind(0)]
            pr_faces = [smpl_faces] * n_humans_i

            if compute_vsmpl_metric:
                # World-space penetration query: which scene points lie inside any SMPL volume?
                verts_world_dev = geotrf(pr_poses_dev[f_id], smpl_out['smpl_v3d'].unsqueeze(0))[0]
                joints_world_dev = geotrf(pr_poses_dev[f_id], smpl_out['smpl_j3d'].unsqueeze(0))[0]
                points_world_dev = pts3ds_other_dev[f_id].reshape(-1, 3)
                # expand() gives a zero-stride view across humans. The volume's internal F.pad
                # allocates a fresh contiguous tensor anyway, so .contiguous() here would only
                # double-allocate without saving downstream work.
                points_world_b = points_world_dev.unsqueeze(0).expand(n_humans_i, -1, -1)
                smpl_volume_state = SimpleNamespace(
                    full_pose=smpl_rotvec_dev[f_id].reshape(n_humans_i, -1),
                    joints=joints_world_dev,
                    vertices=verts_world_dev,
                )
                with torch.no_grad():
                    smpl_layer.bm_x.volume.detach_cache()
                    sdf = smpl_layer.bm_x.volume.query(points_world_b, smpl_volume_state)
                inside = (sdf < 0).any(dim=0)  # [H*W] — true if any body's SDF says inside
                human_mask = msks_dev[f_id].reshape(-1) > 0.5  # pixels labeled as human
                scene_mask = ~human_mask
                n_total = inside.numel()
                n_inside_total = int(inside.sum().item())
                n_inside_scene = int((inside & scene_mask).sum().item())
                n_scene = int(scene_mask.sum().item())
                print(
                    f"[VolumetricSMPL] Frame {f_id:06d}: "
                    f"scene-only {n_inside_scene}/{n_scene} inside, "
                    f"all {n_inside_total}/{n_total} inside, "
                    f"sdf min/med/max={sdf.min().item():.3f}/{sdf.median().item():.3f}/{sdf.max().item():.3f}"
                )
                inside_masks.append(inside.reshape(H, W).cpu())
        else:
            pr_verts = []
            pr_faces = []
            all_verts.append(torch.empty(0))
            if compute_vsmpl_metric:
                inside_masks.append(torch.zeros(H, W, dtype=torch.bool))
                print(f"[VolumetricSMPL] Frame {f_id:06d}: no humans detected")

        if render:
            hm = vis_heatmap(colors_tosave[f_id], smpl_scores[f_id]).numpy()
            img_array_np = (color * 255).astype(np.uint8)
            smpl_rend = render_meshes(img_array_np.copy(), pr_verts, pr_faces,
                                        {'focal': intrins[[0,1],[0,1]], 
                                        'princpt': intrins[[0,1],[-1,-1]]},
                                        color=[get_color(i)/255 for i in smpl_id[f_id]])
            if has_mask:
                msk_array_np = vis_heatmap(colors_tosave[f_id], msks[f_id][0]).numpy()
                color_smpl = np.concatenate([
                    img_array_np, 
                    (msk_array_np * 255).astype(np.uint8), 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
            else:
                color_smpl = np.concatenate([
                    img_array_np, 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
        
        if save:
            np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
            np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
            iio.imwrite(
                os.path.join(outdir, "color", f"{f_id:06d}.png"),
                (color * 255).astype(np.uint8),
            )
            np.savez(
                os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
                pose=c2w,
                intrinsics=intrins,
            )
            np.savez(
                os.path.join(outdir, "smpl", f"{f_id:06d}.npz"),
                scores=smpl_scores[f_id].numpy(),
                msk=msks[f_id].numpy() if has_mask else None,
                shape=smpl_shape[f_id].numpy(),
                rotvec=smpl_rotvec[f_id].numpy(),
                transl=smpl_transl[f_id].numpy(),
                expression=smpl_expression[f_id].numpy() if smpl_expression[f_id] is not None else None
            )

        # Save smpl projection
        if render:
            os.makedirs(os.path.join(outdir, "color_smpl"), exist_ok=True)
            iio.imwrite(
                os.path.join(outdir, "color_smpl", f"{f_id:06d}.png"),
                color_smpl,
            )

    if render and render_video:
        print(f"Saving smpl mesh projection to {outdir}...")
        frames_dir = os.path.join(outdir, "color_smpl")
        video_path = os.path.join(outdir, "output_video.mp4")
        output_fps = 30 // subsample
        os.system(f'/usr/bin/ffmpeg -y -framerate {output_fps} -i "{frames_dir}/%06d.png" '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                f'-movflags +faststart -b:v 5000k "{video_path}"')
    
    return (
        pts3ds_other,
        colors,
        conf_other,
        cam_dict,
        all_verts,
        smpl_faces,
        smpl_id,
        msks,
        inside_masks,
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
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        collate_fn=lambda x: x,
    )
    timings["build_dataset"] = time.time() - t0

    # Run inference, streaming one frame at a time via forward_step. Outputs
    # still accumulate into a single dict for the existing prepare_output;
    # iteration 2 will switch to per-chunk processing.
    # no_grad + autocast(enabled=False) match the wrapping that
    # inference_recurrent_lighter used to provide (inference.py).
    print("Running inference (streaming)...")
    start_time = time.time()
    state_args = None
    trackers = None
    all_pred = []
    all_views = []
    n_processed = 0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
        for view in _frame_iter_from_loader(loader):
            res, state_args, trackers = model.forward_step(
                view, device,
                state_args=state_args, trackers=trackers,
                use_ttt3r=args.use_ttt3r,
            )
            all_pred.append(res)
            all_views.append(_strip_view_for_output(view))
            n_processed += 1
            if n_processed % args.chunk_size == 0:
                print(f"  inference: {n_processed} frames")
    outputs = dict(pred=all_pred, views=all_views)
    total_time = time.time() - start_time
    timings["run_inference"] = total_time
    per_frame_time = total_time / max(n_processed, 1)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    t0 = time.time()
    (
        pts3ds_other, 
        colors,
        conf,
        cam_dict,
        all_smpl_verts,
        smpl_faces,
        smpl_id,
        msks,
        inside_masks,
        ) = prepare_output(
        outputs, args.output_dir, 1, True,
        args.save, args.render, args.render_video, img_res, args.subsample,
        device=device, compute_vsmpl_metric=args.eval_vsmpl,
    )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    msks_to_vis = [m.cpu().numpy() for m in msks]
    conf_to_vis = [c.cpu().numpy() for c in conf]
    edge_colors = [None] * len(pts3ds_to_vis)
    verts_to_vis = [p.cpu().numpy() for p in all_smpl_verts]

    if args.eval_vsmpl:
        # Paint points inside an SMPL volume in red. colors_to_vis[i]: [1, H, W, 3] in [0,1].
        red = np.array([1.0, 0.0, 0.0], dtype=colors_to_vis[0].dtype)
        for i, m in enumerate(inside_masks):
            colors_to_vis[i][0][m.cpu().numpy()] = red

    timings["process_outputs"] = time.time() - t0

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
            "smpl_faces": smpl_faces,
            "smpl_id": [t.cpu().numpy() for t in smpl_id],
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
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
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
