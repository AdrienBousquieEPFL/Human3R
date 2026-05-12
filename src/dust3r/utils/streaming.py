"""Shared streaming helpers used by demo.py and eval/global_human/launch.py.

Both pipelines drive `model.forward_step` over chunks of frames coming from a
torch DataLoader, and both maintain a running cumulative camera-pose base
across chunks. The two helpers below capture the pieces that are truly
identical — the per-pipeline FrameDatasets and ChunkProcessors stay separate
because their fields and output contracts differ.
"""

from copy import deepcopy

import torch


def frame_iter_from_loader(loader):
    """Yield each loaded view, then yield a deepcopy overlap-view (with
    reset=False) immediately after any view whose reset==True.

    Reproduces the overlap-view side effect of the old prepare_input loops,
    so the inference path sees the same frame stream regardless of whether
    the dataset yields one frame at a time or in bulk.

    The overlap copy is taken BEFORE yielding the original view, because
    the caller may mutate the view (e.g., the eval pipeline's
    `smpl_model.update_smpl_gt_eval` pops original smpl_* keys and replaces
    them with derived ones). Deepcopying afterwards would propagate those
    mutations into the overlap and break the next inference step.
    """
    for view in loader:
        is_reset = bool(view["reset"].item())
        if is_reset:
            overlap = deepcopy(view)
            overlap["reset"] = torch.tensor(False).unsqueeze(0)
        yield view
        if is_reset:
            yield overlap


def streaming_pose_recovery(raw_poses, reset_mask, pose_base, any_reset_seen):
    """Streaming version of the matrix_cumprod pose-recovery branch.

    Replicates the byte-exact behavior of the original one-shot
    `prepare_output` two-branch logic across chunks:
      - when no reset has ever been seen and the current chunk has none,
        returns `raw_poses` unchanged (bit-exact with the no-reset code path);
      - otherwise applies the per-frame `cur_base @ raw_poses[i]` matmul,
        advancing `cur_base` only at frames where `reset_mask[i]` is True.

    Inputs:
      raw_poses:       list of [1, 4, 4] tensors (one per kept frame)
      reset_mask:      tensor of length len(raw_poses); per-frame reset flag
      pose_base:       [4, 4] tensor — running cumulative base from prior chunks
      any_reset_seen:  bool — has any reset fired anywhere in the sequence so far

    Returns:
      pr_poses:           list of [1, 4, 4] tensors (length = len(raw_poses))
      new_pose_base:      [4, 4] tensor (unchanged in no-reset path)
      new_any_reset_seen: bool
    """
    chunk_has_reset = bool(reset_mask.any())
    if not (any_reset_seen or chunk_has_reset):
        return raw_poses, pose_base, any_reset_seen
    cur_base = pose_base.clone()
    out = []
    for i, rp in enumerate(raw_poses):
        rp_44 = rp.squeeze(0)
        out.append((cur_base @ rp_44).unsqueeze(0))
        if reset_mask[i]:
            cur_base = cur_base @ rp_44
    return out, cur_base, True
