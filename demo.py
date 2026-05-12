#!/usr/bin/env python3
"""
Modified from CUT3R: https://github.com/CUT3R/CUT3R

Online Human-Scene Reconstruction Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D scene point clouds and SMPLX sequences with the SceneHumanViewer.
Use the command-line arguments to adjust parameters such as the model checkpoint
path, image sequence directory, image size, device, etc.

Example:
    python demo.py --model_path src/human3r_896L.pth --size 512 \
        --seq_path examples/GoodMornin1.mp4 --subsample 1 --vis_threshold 2 \
        --downsample_factor 1 --use_ttt3r --reset_interval 100
"""

import argparse
import glob
import os
import random
import shutil
import tempfile
import time

import cv2
import imageio.v2 as iio
import numpy as np
import roma
import torch

from add_ckpt_path import add_path_to_dust3r

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
        default=10000000,
    )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Use TTT3R.",
        default=False,
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
        help="Frames processed per inference chunk.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers used to decode frames lazily.",
    )
    return parser.parse_args()


class FrameDataset(torch.utils.data.Dataset):
    """Load demo frames lazily while preserving the old reset markers."""

    def __init__(self, img_paths, size, model_path, img_res=None, reset_interval=10000000):
        self.img_paths = list(img_paths)
        self.size = size
        self.model_path = model_path
        self.img_res = img_res
        self.reset_interval = reset_interval
        self._K_mhmr = None
        if img_res is not None:
            from dust3r.utils.geometry import get_camera_parameters

            self._K_mhmr = get_camera_parameters(img_res, device="cpu")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        add_path_to_dust3r(self.model_path)
        from src.dust3r.utils.image import load_images, pad_image

        image = load_images([self.img_paths[index]], size=self.size, verbose=False)[0]
        view = {
            "img": image["img"],
            "true_shape": torch.from_numpy(image["true_shape"]),
            "idx": index,
            "instance": str(index),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((index + 1) % self.reset_interval == 0).unsqueeze(0),
        }
        if self.img_res is not None:
            view["img_mhmr"] = pad_image(view["img"], self.img_res)
            view["K_mhmr"] = self._K_mhmr
        return view


def _strip_view_for_output(view):
    return {
        "img": view["img"].detach().cpu(),
        "reset": view["reset"].detach().cpu(),
    }


class ChunkProcessor:
    """Stream output post-processing one chunk at a time."""

    def __init__(
        self,
        outdir,
        save=False,
        render=False,
        render_video=False,
        img_res=None,
        subsample=1,
        use_pose=True,
    ):
        from src.dust3r.post_process import estimate_focal_knowing_depth
        from src.dust3r.utils import SMPL_Layer, render_meshes, vis_heatmap
        from src.dust3r.utils.camera import pose_encoding_to_camera
        from src.dust3r.utils.geometry import geotrf
        from src.dust3r.utils.image import unpad_image
        from src.dust3r.utils.streaming import streaming_pose_recovery
        from viser_utils import get_color

        self._estimate_focal_knowing_depth = estimate_focal_knowing_depth
        self._SMPL_Layer = SMPL_Layer
        self._render_meshes = render_meshes
        self._vis_heatmap = vis_heatmap
        self._pose_encoding_to_camera = pose_encoding_to_camera
        self._geotrf = geotrf
        self._unpad_image = unpad_image
        self._streaming_pose_recovery = streaming_pose_recovery
        self._get_color = get_color

        self.outdir = outdir
        self.save = save
        self.render = render
        self.render_video = render_video
        self.img_res = img_res
        self.subsample = subsample
        self.use_pose = use_pose

        self.pose_base = torch.eye(4)
        self.any_reset_seen = False
        self.prev_chunk_last_reset = False
        self.frame_counter = 0

        self.smpl_layer = None
        self.smpl_faces = None
        self.viewer_pts3ds = []
        self.viewer_colors = []
        self.viewer_conf = []
        self.viewer_verts = []
        self.viewer_smpl_id = []
        self.viewer_msks = []
        self.cam_focal_chunks = []
        self.cam_pp_chunks = []
        self.cam_R_chunks = []
        self.cam_t_chunks = []

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
            type="smplx",
            gender="neutral",
            num_betas=num_betas,
            kid=False,
            person_center="head",
        )
        self.smpl_faces = self.smpl_layer.bm_x.faces

    def process_chunk(self, chunk_pred, chunk_views):
        if not chunk_pred:
            return

        local_reset = torch.cat([view["reset"] for view in chunk_views], 0)
        prev_last_for_next = bool(local_reset[-1].item())
        shifted_reset_mask = torch.cat(
            [torch.tensor([self.prev_chunk_last_reset]), local_reset[:-1]], dim=0
        )
        keep_mask = ~shifted_reset_mask
        chunk_pred = [pred for pred, keep in zip(chunk_pred, keep_mask.tolist()) if keep]
        chunk_views = [view for view, keep in zip(chunk_views, keep_mask.tolist()) if keep]
        reset_mask = local_reset[keep_mask]
        self.prev_chunk_last_reset = prev_last_for_next
        if not chunk_pred:
            return

        pts3ds_self = torch.cat([pred["pts3d_in_self_view"] for pred in chunk_pred], 0)
        pts3ds_other = [pred["pts3d_in_other_view"] for pred in chunk_pred]
        conf_self = [pred["conf_self"] for pred in chunk_pred]
        conf_other = [pred["conf"] for pred in chunk_pred]

        raw_poses = [
            self._pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in chunk_pred
        ]
        pr_poses, self.pose_base, self.any_reset_seen = self._streaming_pose_recovery(
            raw_poses, reset_mask, self.pose_base, self.any_reset_seen
        )

        if self.use_pose:
            transformed_pts3ds_other = []
            for pose, pts_self in zip(pr_poses, pts3ds_self):
                transformed_pts3ds_other.append(self._geotrf(pose, pts_self.unsqueeze(0)))
            pts3ds_other = transformed_pts3ds_other
            conf_other = conf_self

        batch_size, height, width, _ = pts3ds_self.shape
        pp = torch.tensor([width // 2, height // 2], device=pts3ds_self.device).float().repeat(batch_size, 1)
        focal = self._estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        colors = [
            0.5 * (view["img"].permute(0, 2, 3, 1) + 1.0)
            for view in chunk_views
        ]

        cam2world_tosave = torch.cat(pr_poses)
        intrinsics_tosave = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics_tosave[:, 0, 0] = focal.detach()
        intrinsics_tosave[:, 1, 1] = focal.detach()
        intrinsics_tosave[:, 0, 2] = pp[:, 0]
        intrinsics_tosave[:, 1, 2] = pp[:, 1]

        smpl_shape = [
            pred.get("smpl_shape", torch.empty(1, 0, 10))[0] for pred in chunk_pred
        ]
        smpl_rotvec = [
            roma.rotmat_to_rotvec(pred.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0])
            for pred in chunk_pred
        ]
        smpl_transl = [
            pred.get("smpl_transl", torch.empty(1, 0, 3))[0] for pred in chunk_pred
        ]
        smpl_expression = [
            pred.get("smpl_expression", [None])[0] for pred in chunk_pred
        ]
        smpl_id = [pred.get("smpl_id", torch.empty(1, 0))[0] for pred in chunk_pred]

        if self.render or self.save:
            smpl_scores = [
                pred.get("smpl_scores", torch.zeros(1, height, width, 1))[..., 0]
                for pred in chunk_pred
            ]
            if self.img_res is not None:
                smpl_scores = [self._unpad_image(score, [height, width])[0] for score in smpl_scores]

        has_mask = "msk" in chunk_pred[0]
        if has_mask:
            msks = [pred["msk"][..., 0] for pred in chunk_pred]
            if self.img_res is not None:
                msks = [self._unpad_image(mask, [height, width]) for mask in msks]
        else:
            msks = [torch.zeros(1, height, width) for _ in range(batch_size)]

        self._ensure_smpl_layer(smpl_shape[0].shape[-1])

        depths_tosave = pts3ds_self[..., 2]
        pts3ds_other_tosave = torch.cat(pts3ds_other)
        conf_self_tosave = torch.cat(conf_self)
        conf_other_tosave = torch.cat(conf_other)
        colors_tosave = torch.cat(colors)

        for frame_idx in range(batch_size):
            global_frame_idx = self.frame_counter + frame_idx
            n_humans = smpl_shape[frame_idx].shape[0]

            if n_humans > 0:
                with torch.no_grad():
                    smpl_out = self.smpl_layer(
                        smpl_rotvec[frame_idx],
                        smpl_shape[frame_idx],
                        smpl_transl[frame_idx],
                        None,
                        None,
                        K=intrinsics_tosave[frame_idx].expand(n_humans, -1, -1),
                        expression=smpl_expression[frame_idx],
                    )
                verts_world = self._geotrf(
                    pr_poses[frame_idx], smpl_out["smpl_v3d"].unsqueeze(0)
                )[0]
                pr_verts = [vert.numpy() for vert in smpl_out["smpl_v3d"].unbind(0)]
                pr_faces = [self.smpl_faces] * n_humans
            else:
                verts_world = torch.empty(0)
                pr_verts = []
                pr_faces = []

            depth = depths_tosave[frame_idx].numpy()
            conf = conf_self_tosave[frame_idx].numpy()
            color = colors_tosave[frame_idx].numpy()
            c2w = cam2world_tosave[frame_idx].numpy()
            intrins = intrinsics_tosave[frame_idx].numpy()

            if self.render:
                hm = self._vis_heatmap(colors_tosave[frame_idx], smpl_scores[frame_idx]).numpy()
                img_array_np = (color * 255).astype(np.uint8)
                smpl_rend = self._render_meshes(
                    img_array_np.copy(),
                    pr_verts,
                    pr_faces,
                    {"focal": intrins[[0, 1], [0, 1]], "princpt": intrins[[0, 1], [-1, -1]]},
                    color=[self._get_color(i) / 255 for i in smpl_id[frame_idx]],
                )
                if has_mask:
                    msk_array_np = self._vis_heatmap(colors_tosave[frame_idx], msks[frame_idx][0]).numpy()
                    color_smpl = np.concatenate(
                        [
                            img_array_np,
                            (msk_array_np * 255).astype(np.uint8),
                            (hm * 255).astype(np.uint8),
                            smpl_rend,
                        ],
                        1,
                    )
                else:
                    color_smpl = np.concatenate(
                        [img_array_np, (hm * 255).astype(np.uint8), smpl_rend], 1
                    )

            if self.save:
                np.save(os.path.join(self.outdir, "depth", f"{global_frame_idx:06d}.npy"), depth)
                np.save(os.path.join(self.outdir, "conf", f"{global_frame_idx:06d}.npy"), conf)
                iio.imwrite(
                    os.path.join(self.outdir, "color", f"{global_frame_idx:06d}.png"),
                    (color * 255).astype(np.uint8),
                )
                np.savez(
                    os.path.join(self.outdir, "camera", f"{global_frame_idx:06d}.npz"),
                    pose=c2w,
                    intrinsics=intrins,
                )
                np.savez(
                    os.path.join(self.outdir, "smpl", f"{global_frame_idx:06d}.npz"),
                    scores=smpl_scores[frame_idx].numpy(),
                    msk=msks[frame_idx].numpy() if has_mask else None,
                    shape=smpl_shape[frame_idx].numpy(),
                    rotvec=smpl_rotvec[frame_idx].numpy(),
                    transl=smpl_transl[frame_idx].numpy(),
                    expression=(
                        smpl_expression[frame_idx].numpy()
                        if smpl_expression[frame_idx] is not None
                        else None
                    ),
                )

            if self.render:
                iio.imwrite(
                    os.path.join(self.outdir, "color_smpl", f"{global_frame_idx:06d}.png"),
                    color_smpl,
                )

            self.viewer_pts3ds.append(pts3ds_other_tosave[frame_idx:frame_idx + 1].numpy())
            self.viewer_colors.append(colors_tosave[frame_idx:frame_idx + 1].numpy())
            self.viewer_conf.append(conf_other_tosave[frame_idx:frame_idx + 1].numpy())
            self.viewer_verts.append(verts_world.numpy())
            self.viewer_smpl_id.append(smpl_id[frame_idx].cpu().numpy())
            self.viewer_msks.append(msks[frame_idx].numpy())

        r_c2w = torch.cat([pose[:, :3, :3] for pose in pr_poses], 0)
        t_c2w = torch.cat([pose[:, :3, 3] for pose in pr_poses], 0)
        self.cam_focal_chunks.append(focal.numpy())
        self.cam_pp_chunks.append(pp.numpy())
        self.cam_R_chunks.append(r_c2w.numpy())
        self.cam_t_chunks.append(t_c2w.numpy())
        self.frame_counter += batch_size

    def finalize(self, render_video=False, subsample=1):
        cam_dict = {
            "focal": np.concatenate(self.cam_focal_chunks, 0),
            "pp": np.concatenate(self.cam_pp_chunks, 0),
            "R": np.concatenate(self.cam_R_chunks, 0),
            "t": np.concatenate(self.cam_t_chunks, 0),
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
            self.smpl_faces,
            self.viewer_smpl_id,
            self.viewer_msks,
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
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    add_path_to_dust3r(args.model_path)

    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.streaming import frame_iter_from_loader
    from viser_utils import SceneHumanViewer

    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]

    print(f"Found {len(img_paths)} images in {args.seq_path}.")

    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    print("Building frame dataset...")
    img_res = getattr(model, "mhmr_img_res", None)
    dataset = FrameDataset(
        img_paths=img_paths,
        size=args.size,
        model_path=args.model_path,
        img_res=img_res,
        reset_interval=args.reset_interval,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    print("Running inference...")
    processor = ChunkProcessor(
        args.output_dir,
        save=args.save,
        render=args.render,
        render_video=args.render_video,
        img_res=img_res,
        subsample=args.subsample,
        use_pose=True,
    )
    start_time = time.time()
    state_args = None
    trackers = None
    chunk_pred = []
    chunk_views = []
    n_steps = 0
    try:
        with torch.no_grad():
            for view in frame_iter_from_loader(loader):
                result, state_args, trackers = model.forward_step(
                    view,
                    device,
                    state_args=state_args,
                    trackers=trackers,
                    use_ttt3r=args.use_ttt3r,
                )
                chunk_pred.append(result)
                chunk_views.append(_strip_view_for_output(view))
                n_steps += 1
                if len(chunk_pred) >= args.chunk_size:
                    processor.process_chunk(chunk_pred, chunk_views)
                    chunk_pred.clear()
                    chunk_views.clear()
            if chunk_pred:
                processor.process_chunk(chunk_pred, chunk_views)
    finally:
        if tmpdirname is not None:
            shutil.rmtree(tmpdirname)

    total_time = time.time() - start_time
    per_frame_time = total_time / max(n_steps, 1)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    (
        pts3ds_to_vis,
        colors_to_vis,
        conf_to_vis,
        cam_dict,
        verts_to_vis,
        smpl_faces,
        smpl_id,
        msks_to_vis,
    ) = processor.finalize(args.render_video, args.subsample)
    edge_colors = [None] * len(pts3ds_to_vis)

    print("Launching Human3R viewer...")
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
    viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    run_inference(args)


if __name__ == "__main__":
    main()
