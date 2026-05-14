# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import numpy as np
import torch
from dust3r.utils.geometry import xy_grid


def _import_attach_volume():
    from VolumetricSMPL import attach_volume
    return attach_volume


def attach_pretrained_volume(parametric_body, device):
    """Attach VolumetricSMPL and load weights on the requested device."""
    attach_volume = _import_attach_volume()
    attach_volume(parametric_body, pretrained=False, device=device)
    checkpoint = (
        "https://github.com/markomih/VolumetricSMPL/blob/dev/models/"
        f"VolumetricSMPL_{parametric_body.volume.model_type}_{parametric_body.gender}.ckpt?raw=true"
    )
    state_dict = torch.hub.load_state_dict_from_url(
        checkpoint,
        progress=True,
        map_location=torch.device(device),
    )
    parametric_body.volume.load_state_dict(state_dict["state_dict"])
    return parametric_body.to(device=device)


def _smplx_volume_output(smplx_layer, pose, shape, expression):
    batch_size = pose.shape[0]
    kwargs = {
        "betas": shape,
        "global_orient": pose[:, 0],
        "body_pose": pose[:, 1:22].flatten(1),
        "left_hand_pose": pose[:, 22:37].flatten(1),
        "right_hand_pose": pose[:, 37:52].flatten(1),
        "jaw_pose": pose[:, 52:53].flatten(1),
        "leye_pose": smplx_layer.leye_pose.repeat(batch_size, 1),
        "reye_pose": smplx_layer.reye_pose.repeat(batch_size, 1),
        "return_full_pose": True,
    }
    if expression is not None:
        kwargs["expression"] = expression.flatten(1)
    else:
        kwargs["expression"] = smplx_layer.expression.repeat(batch_size, 1)
    return smplx_layer(**kwargs)


def volumetric_smpl_selfpen_loss(smplx_layer, pose, shape, expression=None):
    """VolumetricSMPL self-intersection loss for SMPL-X pose post-optimization."""
    smplx_layer.volume.detach_cache()
    smpl_volume_out = _smplx_volume_output(smplx_layer, pose, shape, expression)
    return smplx_layer.volume.self_collision_loss(smpl_volume_out).mean()


def optimize_smpl_selfpen(
    smplx_layer,
    pose,
    shape,
    expression=None,
    steps=10,
    lr=1e-3,
    optimize_device="cpu",
):
    """Post-optimize SMPL-X pose with only the VolumetricSMPL selfpen loss."""
    if steps <= 0 or pose.shape[0] == 0:
        return pose, {"initial_loss": 0.0, "final_loss": 0.0, "steps": 0}

    output_device = pose.device
    original_device = next(smplx_layer.parameters()).device
    optimize_device = torch.device(optimize_device)

    smplx_layer.volume.detach_cache()
    smplx_layer.to(optimize_device)
    opt_pose = pose.detach().to(optimize_device).clone().requires_grad_(True)
    shape = shape.detach().to(optimize_device)
    expression = (
        expression.detach().to(optimize_device) if expression is not None else None
    )
    optimizer = torch.optim.Adam([opt_pose], lr=lr)
    initial_loss = None
    final_loss = None
    n_steps = 0

    was_training = smplx_layer.training
    smplx_layer.eval()
    try:
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = volumetric_smpl_selfpen_loss(
                    smplx_layer, opt_pose, shape, expression
                )
                if initial_loss is None:
                    initial_loss = float(loss.detach().cpu())
                final_loss = float(loss.detach().cpu())
                if not torch.isfinite(loss) or final_loss <= 1e-8:
                    break
                loss.backward()
                optimizer.step()
                n_steps += 1
    finally:
        if was_training:
            smplx_layer.train()
        smplx_layer.volume.detach_cache()
        smplx_layer.to(original_device)

    return opt_pose.detach().to(output_device), {
        "initial_loss": 0.0 if initial_loss is None else initial_loss,
        "final_loss": 0.0 if final_loss is None else final_loss,
        "steps": n_steps,
        "device": str(optimize_device),
    }


def estimate_focal_knowing_depth(
    pts3d, pp, focal_mode="median", min_focal=0.0, max_focal=np.inf
):
    """Reprojection method, for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(
        -1, 1, 2
    )  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == "median":
        with torch.no_grad():

            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == "weiszfeld":

        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(
            posinf=0, neginf=0
        )  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        for iter in range(10):

            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)

            w = dis.clip(min=1e-8).reciprocal()

            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f"bad {focal_mode=}")

    focal_base = max(H, W) / (
        2 * np.tan(np.deg2rad(60) / 2)
    )  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)

    return focal
