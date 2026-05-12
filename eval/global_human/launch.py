import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.global_human.metadata import dataset_metadata
from eval.global_human.utils import *

from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r

from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="bedlam",
        choices=list(dataset_metadata.keys()),
    )

    parser.add_argument("--crop_res", type=int, nargs=2, metavar=("W", "H"), default=None)
    parser.add_argument("--size", type=int, default="224")
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument("--solve_pose", action="store_true", default=False)
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--is_naive", action="store_true", default=False)
    parser.add_argument("--use_ttt3r", action="store_true", default=False)
    parser.add_argument("--reset_interval", type=int, default=100000000)
    parser.add_argument("--use_fake_K", action="store_true", default=False)
    parser.add_argument(
        "--chunk_size", type=int, default=32,
        help="Frames per inference chunk; bounds peak inference RAM.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader workers for input frame decoding. 0 keeps everything in main process.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="If set, truncate each sequence's filelist to this many frames "
             "(verification convenience; leave unset for normal runs).",
    )
    return parser

def get_seq_list(metadata, img_path):
    get_seq_func = metadata.get("get_seq_func", None)
    split = metadata.get("split", "")
    
    if get_seq_func:
        annots = metadata["get_annot_func"](img_path, split)
        seq_list, seq_to_images = get_seq_func(img_path, split, annots)
        return seq_list, seq_to_images, annots
    
    if metadata.get("full_seq", False):
        seq_dir = f"{img_path}/{split}"
        seq_list = [seq for seq in os.listdir(seq_dir) 
                   if os.path.isdir(os.path.join(seq_dir, seq))]
        return sorted(seq_list), None, None
    else:
        return  sorted(metadata.get("seq_list", [])), None, None

def get_file_list(metadata, img_path, seq, seq_to_images=None):
    dir_path = metadata["dir_path_func"](img_path, seq)
    subsample = metadata.get("subsample", 1)
    max_frames = metadata.get("max_frames", None)

    if seq_to_images is not None:
        filelist = [os.path.join(dir_path, name) for name in seq_to_images[seq]]
    else:
        filelist = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]

    filelist.sort()
    
    if max_frames is not None:
        filelist = filelist[:max_frames]

    sampled_indices = list(range(0, len(filelist), subsample))
    filelist = filelist[::subsample]
    return filelist, sampled_indices

def get_pred_smpl(pred, gt, f_id, is_naive, smpl_layer, mhmr_img_res, K_to_proj):
    n_humans_i = pred['shape'][f_id].shape[0]
    expand = lambda x: x.expand(n_humans_i, -1, -1)

    with torch.no_grad():
        if is_naive:
            dist = pred['transl'][f_id][:, 0].unsqueeze(-1)
            dist = to_euclidean_dist(
                mhmr_img_res, dist, expand(gt['K_mhmr'][f_id]))
            smpl_out = smpl_layer(
                pred['rotvec'][f_id], 
                pred['shape'][f_id], 
                None, 
                pred['loc'][f_id], 
                dist, 
                K=expand(gt['K_mhmr'][f_id]), 
                expression=pred['expression'][f_id],
                K_to_proj=expand(gt['K'][f_id]),
                )
            pred['transl'][f_id] = smpl_out['smpl_transl']
        else:
            smpl_out = smpl_layer(
                pred['rotvec'][f_id], 
                pred['shape'][f_id], 
                pred['transl'][f_id], 
                None, None, 
                K=expand(K_to_proj[f_id]), 
                expression=pred['expression'][f_id])
    
    return smpl_out['smpl_v3d']

def match_2d(pr_j2d, gt_j2d):
    # match pred to gt - based on 2d bbox
    gt_j2d = gt_j2d.numpy()
    bestMatch, falsePositives, misses = match_2d_greedy(
        pr_j2d.numpy()[:,:gt_j2d.shape[1]],
        gt_j2d,
        np.ones_like(gt_j2d[...,0]).astype(np.bool_))

    update = {
        'count': len(gt_j2d),
        'miss': len(misses),
        'fp': len(falsePositives)
    }

    return bestMatch, update


class EvalChunkProcessor:
    """Streams the per-sequence post-processing + metric computation one chunk
    at a time. Carries minimal state across chunks (cumulative camera-pose
    base, overlap-drop boundary flag, frame counter, accumulating
    counter/metrics/global_batch). The chunk's fp32 pred/gt tensors are
    freed before the next chunk runs.

    Mirrors the body of the previous prepare_output + the per-frame metric
    body in eval_smpl_error (launch.py:308-385) byte-for-byte when the
    streaming pose recovery hits the no-reset path (which is the common
    case at default reset_interval).
    """

    def __init__(
        self, args, smpl_model, smpl_layer, pelvis_idx, mhmr_img_res,
        is_global, save_dir, seq, subsample, filelist=None,
    ):
        from dust3r.utils.streaming import streaming_pose_recovery
        from dust3r.utils.camera import pose_encoding_to_camera
        from dust3r.post_process import estimate_focal_knowing_depth
        from dust3r.utils.image import unpad_image
        from dust3r.utils.geometry import perspective_projection, geotrf

        self._streaming_pose_recovery = streaming_pose_recovery
        self._pose_encoding_to_camera = pose_encoding_to_camera
        self._estimate_focal_knowing_depth = estimate_focal_knowing_depth
        self._unpad_image = unpad_image
        self._perspective_projection = perspective_projection
        self._geotrf = geotrf

        if args.solve_pose:
            raise NotImplementedError(
                "--solve_pose is not supported in streaming mode (procrustes is global)."
            )

        self.args = args
        self.smpl_model = smpl_model
        self.smpl_layer = smpl_layer
        self.pelvis_idx = pelvis_idx
        self.mhmr_img_res = mhmr_img_res
        self.is_global = is_global
        self.save_dir = save_dir
        self.seq = seq
        self.subsample = subsample
        self.filelist = filelist

        # Cross-chunk state.
        self.pose_base = torch.eye(4)
        self.any_reset_seen = False
        self.prev_chunk_last_reset = False
        self.frame_counter = 0

        # Accumulators across chunks.
        self.counter = Counter()
        self.metrics = defaultdict(list)
        self.global_batch = []

    def _process_pred(self, chunk_pred, reset_mask):
        """Per-chunk version of the old prepare_output body. Returns a pred
        dict with the same keys: K, T_c2w (concatenated [B, ...]) and SMPL
        params as per-frame lists."""
        pts3ds_self_to_save = [p["pts3d_in_self_view"] for p in chunk_pred]
        conf_self_ls = [p["conf_self"] for p in chunk_pred]
        pts3ds_self = torch.cat(pts3ds_self_to_save, 0)

        raw_poses = [
            self._pose_encoding_to_camera(p["camera_pose"].clone())
            for p in chunk_pred
        ]  # list of [1, 4, 4]
        # Streaming pose recovery: when no reset has fired anywhere, raw_poses
        # are returned unchanged (bit-exact with the original no-reset branch).
        pr_poses, self.pose_base, self.any_reset_seen = self._streaming_pose_recovery(
            raw_poses, reset_mask, self.pose_base, self.any_reset_seen,
        )
        pr_poses_cat = torch.cat(pr_poses, 0)  # [B, 4, 4]

        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 2)
        )
        focal = self._estimate_focal_knowing_depth(
            pts3ds_self, pp, focal_mode="weiszfeld",
        )

        intrinsics = torch.eye(3, device=pp.device).unsqueeze(0).repeat(B, 1, 1)
        intrinsics[:, 0, 0] = focal
        intrinsics[:, 1, 1] = focal
        intrinsics[:, [0, 1], 2] = pp

        pred = {}
        pred['T_c2w'] = pr_poses_cat
        pred['K'] = intrinsics
        pred['shape'] = [
            p.get("smpl_shape", torch.empty(1, 0, 10))[0] for p in chunk_pred
        ]
        pred['rotvec'] = [
            roma.rotmat_to_rotvec(
                p.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0]
            )
            for p in chunk_pred
        ]
        pred['transl'] = [
            p.get("smpl_transl", torch.empty(1, 0, 3))[0] for p in chunk_pred
        ]
        pred['expression'] = [
            p.get("smpl_expression", [None])[0] for p in chunk_pred
        ]
        pred['loc'] = [
            p.get("smpl_loc", torch.empty(1, 0, 2))[0] for p in chunk_pred
        ]

        if self.args.save:
            has_mask = "msk" in chunk_pred[0]
            if has_mask:
                msks = [p["msk"][..., 0] for p in chunk_pred]
                msks = [self._unpad_image(m, [H, W]) for m in msks]
            else:
                msks = [torch.zeros(1, H, W) for _ in range(B)]
            pred['pts3d_self'] = pts3ds_self_to_save
            pred['conf_self'] = conf_self_ls
            pred['msk'] = msks

        return pred

    def _process_gt(self, chunk_views):
        """Per-chunk version of the old prepare_gt body."""
        gt = defaultdict(list)
        intrinsics = [v["camera_intrinsics"] for v in chunk_views]
        K_mhmr = [v["K_mhmr"] for v in chunk_views]
        camera_pose = [v["camera_pose"] for v in chunk_views]
        imgs = [v["img"] for v in chunk_views]
        gt['K'] = torch.cat(intrinsics, 0)
        gt['K_mhmr'] = torch.cat(K_mhmr, 0)
        gt['T_c2w'] = torch.cat(camera_pose, 0)
        gt['img'] = torch.cat(imgs, 0)
        if 'T_w2c' in chunk_views[0]:
            T_w2c_list = [v["T_w2c"] for v in chunk_views]
            gt['T_w2c'] = torch.cat(T_w2c_list, 0)
        for v in chunk_views:
            smpl_mask = v["smpl_mask"]
            gt['v3d_c'].append(v["smpl_v3d_c"][smpl_mask])
            gt['j3d_c'].append(v["smpl_j3d_c"][smpl_mask])
            gt['v3d_w'].append(v["smpl_v3d_w"][smpl_mask])
            gt['j3d_w'].append(v["smpl_j3d_w"][smpl_mask])
            gt['v2d'].append(v["smpl_v2d"][smpl_mask])
            gt['j2d'].append(v["smpl_j2d"][smpl_mask])
        return gt

    def process_chunk(self, chunk_pred, chunk_views):
        """Drop overlap views at the chunk boundary, build pred + gt for the
        chunk, run the per-frame metric body, accumulate counter / metrics /
        global_batch."""
        if not chunk_pred:
            return

        # Step A — cross-chunk overlap drop. Mirrors the original
        # prepare_output's `shifted_reset_mask` logic, with the leading edge
        # carried across chunks via self.prev_chunk_last_reset.
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
            return

        B = len(chunk_pred)

        # Step B — build pred + gt for this chunk.
        pred = self._process_pred(chunk_pred, reset_mask)
        gt = self._process_gt(chunk_views)

        # Step C — per-frame metric loop (mirrors launch.py:311-385).
        K_to_proj = gt['K'] if self.args.is_naive else pred['K']
        T_c2w = pred['T_c2w']

        for f_id_local in range(B):
            f_id_global = self.frame_counter + f_id_local
            n_humans_i = pred['shape'][f_id_local].shape[0]

            if n_humans_i > 0:
                pred_v3d_c = get_pred_smpl(
                    pred, gt, f_id_local, self.args.is_naive,
                    self.smpl_layer, self.mhmr_img_res, K_to_proj,
                )
                pred_v3d_c = self.smpl_model.smplx2smpl @ pred_v3d_c
                pred_j3d_c = self.smpl_model.j_regressor @ pred_v3d_c
                pr_j2d = self._perspective_projection(
                    pred_j3d_c, K_to_proj[f_id_local].expand(n_humans_i, -1, -1),
                )
            else:
                pred_v3d_c = torch.empty(0, 6890, 3, dtype=torch.float32)
                pr_j2d = torch.empty(0, 24, 2, dtype=torch.float32)

            bestMatch, update = match_2d(pr_j2d, gt['j2d'][f_id_local])
            self.counter.update(update)

            if len(bestMatch) > 0:
                self.counter.update({'n_human': len(bestMatch)})
                pid, gid = bestMatch[:, 0], bestMatch[:, 1]

                camcoord_batch = {
                    "pred_j3d": pred_j3d_c[pid],
                    "target_j3d": gt['j3d_c'][f_id_local][gid],
                    "pred_v3d": pred_v3d_c[pid],
                    "target_v3d": gt['v3d_c'][f_id_local][gid],
                }
                camcoord_metrics = eval_camcoord(camcoord_batch, self.pelvis_idx)
                for k, v in camcoord_metrics.items():
                    self.metrics[k].append(v)

                if self.is_global:
                    expand = lambda x: x.expand(len(bestMatch), -1, -1)
                    self.global_batch.append({
                        "pred_j3d": self._geotrf(
                            expand(T_c2w[f_id_local]), pred_j3d_c[pid],
                        ),
                        "target_j3d": gt['j3d_w'][f_id_local][gid],
                        "pred_v3d": self._geotrf(
                            expand(T_c2w[f_id_local]), pred_v3d_c[pid],
                        ),
                        "target_v3d": gt['v3d_w'][f_id_local][gid],
                    })

            if self.args.save:
                color = 0.5 * (gt['img'][f_id_local].permute(1, 2, 0) + 1.0)
                out_dir = f"{self.save_dir}/{self.seq}/{f_id_global:06d}"
                for k in ["pts3d", "conf", "color", "camera", "smpl", "mask"]:
                    os.makedirs(os.path.join(out_dir, k), exist_ok=True)
                np.save(
                    os.path.join(out_dir, "pts3d", f"{f_id_global:06d}.npy"),
                    pred['pts3d_self'][f_id_local],
                )
                np.save(
                    os.path.join(out_dir, "conf", f"pred_{f_id_global:06d}.npy"),
                    pred['conf_self'][f_id_local],
                )
                np.save(
                    os.path.join(out_dir, "color", f"{f_id_global:06d}.npy"), color,
                )
                np.save(
                    os.path.join(out_dir, "mask", f"pred_{f_id_global:06d}.npy"),
                    pred['msk'][f_id_local],
                )
                np.savez(
                    os.path.join(out_dir, "camera", f"pred_{f_id_global:06d}.npz"),
                    pose=pred['T_c2w'][f_id_local], K=pred['K'][f_id_local],
                )
                np.savez(
                    os.path.join(out_dir, "camera", f"gt_{f_id_global:06d}.npz"),
                    pose=gt['T_c2w'][f_id_local], K=gt['K'][f_id_local],
                )
                if len(bestMatch) > 0:
                    np.save(
                        os.path.join(out_dir, "smpl", f"pred_{f_id_global:06d}.npy"),
                        pred_v3d_c[pid],
                    )
                    np.save(
                        os.path.join(out_dir, "smpl", f"gt_{f_id_global:06d}.npy"),
                        gt['v3d_c'][f_id_local][gid],
                    )
                else:
                    np.save(
                        os.path.join(out_dir, "smpl", f"pred_{f_id_global:06d}.npy"),
                        pred_v3d_c,
                    )
                    np.save(
                        os.path.join(out_dir, "smpl", f"gt_{f_id_global:06d}.npy"),
                        gt['v3d_c'][f_id_local],
                    )

            if self.args.vis:
                img_path = (
                    self.filelist[f_id_global]
                    if self.filelist is not None and f_id_global < len(self.filelist)
                    else None
                )
                visualize(
                    save_dir=f"{self.save_dir}/{self.seq}",
                    img_path=img_path,
                    view=gt['img'][f_id_local],
                    gt_v3d_c=gt['v3d_c'][f_id_local],
                    pred_v3d_c=pred_v3d_c,
                    K_to_proj=K_to_proj[f_id_local],
                    gt_K=gt['K'][f_id_local],
                    bestMatch=bestMatch,
                    smpl_face=self.smpl_model.smpl_faces['smpl'],
                )

        self.frame_counter += B

    def finalize(self):
        """Compute precision/recall/F1 and (if is_global) eval_global on the
        accumulated buffer. Returns (counter, metrics) for write_log."""
        self.metrics['precision'], self.metrics['recall'], self.metrics['f1_score'] = compute_prf1(
            self.counter['count'], self.counter['miss'], self.counter['fp'],
        )
        if self.is_global and self.global_batch:
            global_batch_cat = {
                k: torch.cat([b[k] for b in self.global_batch])
                for k in self.global_batch[0]
            }
            global_metrics = eval_global(global_batch_cat, self.subsample)
            for k, v in global_metrics.items():
                self.metrics[k].append(v)
        return self.counter, self.metrics


def eval_smpl_error(args, model, smpl_model, smpl_layer, save_dir=None):
    from dust3r.utils.streaming import frame_iter_from_loader

    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mhmr_img_res = getattr(model, "mhmr_img_res", None)
    subsample = metadata.get("subsample", 1)
    is_global = metadata["is_global"](metadata.get("split", ""))
    pelvis_idx = smpl_model.pelvis_idx

    seq_list, seq_to_images, annots = get_seq_list(metadata, img_path)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    keep_keys = {"img", "img_mask", "true_shape", "img_mhmr", "reset", "update", "K_mhmr"}

    with distributed_state.split_between_processes(seq_list) as seqs:
        if len(seq_list) < distributed_state.num_processes:
            if distributed_state.process_index >= len(seq_list):
                seqs = []
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"

        for seq_idx, seq in enumerate(tqdm(seqs)):
            try:
                print(f"Evaluating sequence: {seq}")
                filelist, sampled_indices = get_file_list(metadata, img_path, seq, seq_to_images)
                if args.max_frames is not None:
                    filelist = filelist[:args.max_frames]
                    sampled_indices = sampled_indices[:args.max_frames]

                # Build per-sequence dataset, loader, and chunk processor.
                get_view_func = metadata.get("get_view_func", None)
                mask_path_func = metadata.get("mask_path_func", None)
                mask_path_list = mask_path_func(filelist) if mask_path_func is not None else []

                dataset = EvalFrameDataset(
                    img_paths=filelist,
                    msk_paths=mask_path_list,
                    size=args.size,
                    crop=not args.no_crop,
                    load_func=get_view_func,
                    annots=annots,
                    sampled_indices=sampled_indices,
                    reset_interval=args.reset_interval,
                    crop_res=args.crop_res,
                )
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=None, num_workers=args.num_workers,
                    pin_memory=True, persistent_workers=(args.num_workers > 0),
                    collate_fn=lambda x: x,
                )

                processor = EvalChunkProcessor(
                    args=args, smpl_model=smpl_model, smpl_layer=smpl_layer,
                    pelvis_idx=pelvis_idx, mhmr_img_res=mhmr_img_res,
                    is_global=is_global, save_dir=save_dir, seq=seq,
                    subsample=subsample, filelist=filelist,
                )

                # Streaming inference + chunk processing. Per-frame: enrich GT
                # via update_smpl_gt_eval, strip to inference-only keys, run
                # forward_step, accumulate one chunk's worth, flush.
                state_args, trackers = None, None
                chunk_pred, chunk_views = [], []
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                    for view in frame_iter_from_loader(loader):
                        smpl_model.update_smpl_gt_eval([view], args.eval_dataset)
                        inf_view = {k: view[k] for k in keep_keys if k in view}
                        res, state_args, trackers = model.forward_step(
                            inf_view, device,
                            state_args=state_args, trackers=trackers,
                            use_ttt3r=args.use_ttt3r,
                        )
                        chunk_pred.append(res)
                        chunk_views.append(view)
                        if len(chunk_pred) >= args.chunk_size:
                            processor.process_chunk(chunk_pred, chunk_views)
                            chunk_pred.clear()
                            chunk_views.clear()
                            gc.collect()
                            torch.cuda.empty_cache()
                    if chunk_pred:
                        processor.process_chunk(chunk_pred, chunk_views)
                        chunk_pred.clear()
                        chunk_views.clear()

                counter, metrics = processor.finalize()

                torch.cuda.empty_cache()

                # Write to error log after each sequence
                os.makedirs(save_dir, exist_ok=True)
                write_log(error_log_path, args.eval_dataset, seq, counter, metrics)

                del processor, dataset, loader, counter, metrics
                del filelist, sampled_indices, state_args, trackers
                gc.collect()


            except Exception as e:
                print(f"Exception in sequence {seq}: {str(e)}")
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e

    distributed_state.wait_for_everyone()
    torch.cuda.empty_cache()

    results = process_directory(save_dir)
    summary = calculate_averages(results)

    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())

            log = get_summary_log(summary)
            f.write(log) 
    
        print(log.strip())
        


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.utils.image import load_masks_for_eval as load_masks
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import weighted_procrustes, to_euclidean_dist, matrix_cumprod
    from dust3r.smpl_model import SMPLModel
    from dust3r.utils import SMPL_Layer
    from dust3r.utils.image import unpad_image

    args.no_crop = False

    def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 1, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        pts3ds_self = pts3ds_self.reshape(B, -1, 3)
        pts3ds_other = pts3ds_other.reshape(B, -1, 3)
        conf_self = conf_self.reshape(B, -1)
        conf_other = conf_other.reshape(B, -1)
        # weighted procrustes
        c2w = weighted_procrustes(
            pts3ds_self,
            pts3ds_other,
            torch.log(conf_self) * torch.log(conf_other),
            use_weights=True,
            return_T=True,
        )
        return c2w, focal, pp.reshape(B, 2)
    
    def _crop_resize(image, intrinsics, crop_res):
        import dust3r.datasets.utils.cropping as cropping
        from dust3r.utils.image import ImgNorm
        
        # image is a tensor in CHW with values in [-1, 1]; convert to HWC uint8 for PIL
        img_device = image.device if isinstance(image, torch.Tensor) else None
        had_batch_dim = False
        if isinstance(image, torch.Tensor):
            # accept [3,H,W] or [1,3,H,W]; squeeze batch if present
            if image.dim() == 4:
                assert image.shape[0] == 1, "_crop_resize expects a single image; got a batch"
                image = image.squeeze(0)
                had_batch_dim = True
            elif image.dim() != 3:
                raise RuntimeError(f"Unexpected image tensor shape {tuple(image.shape)}; expected CHW or 1xCHW")

            image_np = (
                (image.detach().cpu().permute(1, 2, 0).numpy() * 0.5 +0.5) * 255
            ).clip(0, 255).astype(np.uint8)
        else:
            image_np = image

        target_resolution = np.array(crop_res)
        image_pil, _, intrinsics = cropping.rescale_image_depthmap(
            image_np, None, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image_pil.size, crop_res, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, crop_res
        )
        image_pil, _, intrinsics2 = cropping.crop_image_depthmap(
            image_pil, None, intrinsics, crop_bbox
        )

        # convert back to normalized CHW tensor on original device
        image_arr = np.array(image_pil)
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., None], 3, axis=2)
        image_tensor = ImgNorm(image_pil)  # CHW, [-1, 1]
        if had_batch_dim:
            image_tensor = image_tensor.unsqueeze(0)
        if img_device is not None:
            image_tensor = image_tensor.to(img_device)

        # return intrinsics as torch tensor with batch dim like upstream expects
        intrinsics_tensor = torch.from_numpy(intrinsics2).unsqueeze(0)

        return image_tensor, intrinsics_tensor

    class EvalFrameDataset(torch.utils.data.Dataset):
        """Lazily produces one view dict per frame for the eval streaming pipeline.

        Mirrors the no-raymap branch of the previous prepare_input (above), but
        loads one frame at a time via load_images on a single-element list and
        drops the [1, 6, H, W] NaN ray_map placeholder, which is unused on the
        lighter inference path (forward_step / _recurrent_rollout never read it).
        """

        def __init__(
            self, img_paths, msk_paths, size, crop, load_func, annots,
            sampled_indices, reset_interval, crop_res,
        ):
            self.img_paths = list(img_paths)
            self.msk_paths = list(msk_paths) if msk_paths else []
            self.size = size
            self.crop = crop
            self.load_func = load_func
            self.annots = annots
            self.sampled_indices = list(sampled_indices)
            self.reset_interval = reset_interval
            self.crop_res = crop_res

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, i):
            images = load_images(
                [self.img_paths[i]], size=self.size, verbose=False, crop=self.crop,
            )
            images = self.load_func(
                ([self.img_paths[i]], images, self.annots, [self.sampled_indices[i]]),
            )
            image = images[0]

            view = {
                "img": image["img"],
                "true_shape": torch.from_numpy(image["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(image["camera_pose"]).unsqueeze(0),
                "camera_intrinsics": torch.from_numpy(image["intrinsics"]).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i + 1) % self.reset_interval == 0).unsqueeze(0),
            }

            if self.crop_res is not None:
                view["img"], view["camera_intrinsics"] = _crop_resize(
                    image["img"], image["intrinsics"], self.crop_res,
                )
                view["true_shape"] = torch.tensor(
                    [view["img"].shape[-2:]], dtype=torch.int32,
                )

            if self.msk_paths:
                msks = load_masks(
                    [self.msk_paths[i]], size=self.size, verbose=False, crop=self.crop,
                )
                view["msk"] = msks[0]

            for key in image.keys():
                if key.startswith(("smpl", "T_w2c")):
                    view[key] = torch.tensor(image[key]).unsqueeze(0)

            return view

    model = ARCroco3DStereo.from_pretrained(args.weights)
    # SMPL model for gt
    smpl_model = SMPLModel(
        "cpu", 
        model_args={
            'patch_size': model.croco_args['patch_size'], 
            'mhmr_img_res': model.mhmr_img_res, 
            'bb_patch_size': model.bb_patch_size
        },
        eval_args={
            'dataset': args.eval_dataset,
            'use_fake_K': args.use_fake_K
        }
    )
    # SMPL layer for pred
    smpl_layer = SMPL_Layer(type='smplx',
                            gender='neutral',
                            num_betas=10,
                            kid=False, 
                            person_center='head')

    eval_smpl_error(args, model, smpl_model, smpl_layer, save_dir=args.output_dir)
