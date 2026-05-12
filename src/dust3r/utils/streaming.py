from copy import deepcopy

import torch


def frame_iter_from_loader(loader):
    """Yield each frame, inserting the reset-overlap copy used by prepare_input."""
    for view in loader:
        is_reset = bool(view["reset"].item())
        if is_reset:
            overlap = deepcopy(view)
            overlap["reset"] = torch.tensor(False).unsqueeze(0)
        yield view
        if is_reset:
            yield overlap


def streaming_pose_recovery(raw_poses, reset_mask, pose_base, any_reset_seen):
    """Recover camera poses chunk by chunk while matching prepare_output semantics."""
    chunk_has_reset = bool(reset_mask.any())
    if not (any_reset_seen or chunk_has_reset):
        return raw_poses, pose_base, any_reset_seen

    cur_base = pose_base.clone()
    out = []
    for i, raw_pose in enumerate(raw_poses):
        raw_pose_44 = raw_pose.squeeze(0)
        out.append((cur_base @ raw_pose_44).unsqueeze(0))
        if reset_mask[i]:
            cur_base = cur_base @ raw_pose_44
    return out, cur_base, True
