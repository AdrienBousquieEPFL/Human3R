from copy import deepcopy
import cv2

import os
import re
import numpy as np
import torch
import roma
import math
import tqdm
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.transform import Rotation
from eval.relpose.evo_utils import *
from PIL import Image
import imageio.v2 as iio
from matplotlib.figure import Figure
from itertools import product

def todevice(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def to_numpy(x):
    return todevice(x, "numpy")


def compute_prf1(count, miss, fp):
    """
    Code modified from https://github.com/Arthur151/ROMP/blob/4eebd3647f57d291d26423e51f0d514ff7197cb3/simple_romp/evaluation/RH_evaluation/evaluation.py#L90
    """
    if count == 0:
        return 0, 0, 0
    all_tp = count - miss
    all_fp = fp
    all_fn = miss
    if all_tp == 0:
        return 0., 0., 0.
    all_f1_score = round(all_tp / (all_tp + 0.5 * (all_fp + all_fn)), 2)
    all_recall = round(all_tp / (all_tp + all_fn), 2)
    all_precision = round(all_tp / (all_tp + all_fp), 2)
    return 100. * all_precision, 100.* all_recall, 100. * all_f1_score


def match_2d_greedy(
        pred_kps,
        gtkp,
        valid_mask,
        imgPath=None,
        baseline=None,
        iou_thresh=0.05,
        valid=None,
        ind=-1):
    '''
    Code modified from: https://github.com/Arthur151/ROMP/blob/4eebd3647f57d291d26423e51f0d514ff7197cb3/simple_romp/trace2/evaluation/eval_3DPW.py#L232
    matches groundtruth keypoints to the detection by considering all possible matchings.
    :return: best possible matching, a list of tuples, where each tuple corresponds to one match of pred_person.to gt_person.
            the order within one tuple is as follows (idx_pred_kps, idx_gt_kps)
    '''
    predList = np.arange(len(pred_kps))
    gtList = np.arange(len(gtkp))
    # get all pairs of elements in pred_kps, gtkp
    # all combinations of 2 elements from l1 and l2
    combs = list(product(predList, gtList))

    errors_per_pair = {}
    errors_per_pair_list = []
    for comb in combs:
        vmask = valid_mask[comb[1]]
        assert vmask.sum()>0, print('no valid points')
        errors_per_pair[str(comb)] = np.linalg.norm(pred_kps[comb[0]][vmask, :2] - gtkp[comb[1]][vmask, :2], 2)
        errors_per_pair_list.append(errors_per_pair[str(comb)])

    gtAssigned = np.zeros((len(gtkp),), dtype=bool)
    opAssigned = np.zeros((len(pred_kps),), dtype=bool)
    errors_per_pair_list = np.array(errors_per_pair_list)

    bestMatch = []
    excludedGtBecauseInvalid = []
    falsePositiveCounter = 0
    while np.sum(gtAssigned) < len(gtAssigned) and np.sum(
            opAssigned) + falsePositiveCounter < len(pred_kps):
        found = False
        falsePositive = False
        while not(found):
            if sum(np.inf == errors_per_pair_list) == len(
                    errors_per_pair_list):
                print('something went wrong here')

            minIdx = np.argmin(errors_per_pair_list)
            minComb = combs[minIdx]
            # compute IOU
            iou = get_bbx_overlap(
                pred_kps[minComb[0]], gtkp[minComb[1]]) #, imgPath, baseline)
            # if neither prediction nor ground truth has been matched before and iou
            # is larger than threshold
            if not(opAssigned[minComb[0]]) and not(
                    gtAssigned[minComb[1]]) and iou >= iou_thresh:
                #print(imgPath + ': found matching')
                found = True
                errors_per_pair_list[minIdx] = np.inf
            else:
                errors_per_pair_list[minIdx] = np.inf
                # if errors_per_pair_list[minIdx] >
                # matching_threshold*headBboxs[combs[minIdx][1]]:
                if iou < iou_thresh:
                    #print(
                    #   imgPath + ': false positive detected using threshold')
                    found = True
                    falsePositive = True
                    falsePositiveCounter += 1

        # if ground truth of combination is valid keep the match, else exclude
        # gt from matching
        if not(valid is None):
            if valid[minComb[1]]:
                if not falsePositive:
                    bestMatch.append(minComb)
                    opAssigned[minComb[0]] = True
                    gtAssigned[minComb[1]] = True
            else:
                gtAssigned[minComb[1]] = True
                excludedGtBecauseInvalid.append(minComb[1])

        elif not falsePositive:
            # same as above but without checking for valid
            bestMatch.append(minComb)
            opAssigned[minComb[0]] = True
            gtAssigned[minComb[1]] = True

    bestMatch = np.array(bestMatch)
    # add false positives and false negatives to the matching
    # find which elements have been successfully assigned
    opAssigned = []
    gtAssigned = []
    for pair in bestMatch:
        opAssigned.append(pair[0])
        gtAssigned.append(pair[1])
    opAssigned.sort()
    gtAssigned.sort()

    falsePositives = []
    misses = []

    # handle false positives
    opIds = np.arange(len(pred_kps))
    # returns values of oIds that are not in opAssigned
    notAssignedIds = np.setdiff1d(opIds, opAssigned)
    for notAssignedId in notAssignedIds:
        falsePositives.append(notAssignedId)
    gtIds = np.arange(len(gtList))
    # returns values of gtIds that are not in gtAssigned
    notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned)

    # handle false negatives/misses
    for notAssignedIdGt in notAssignedIdsGt:
        if not(valid is None):  # if using the new matching
            if valid[notAssignedIdGt]:
                #print(imgPath + ': miss')
                misses.append(notAssignedIdGt)
            else:
                excludedGtBecauseInvalid.append(notAssignedIdGt)
        else:
            #print(imgPath + ': miss')
            misses.append(notAssignedIdGt)

    return bestMatch, falsePositives, misses  # tuples are (idx_pred_kps, idx_gt_kps)

def get_bbx_overlap(p1, p2):
    """
    Code modifed from https://github.com/Arthur151/ROMP/blob/4eebd3647f57d291d26423e51f0d514ff7197cb3/simple_romp/trace2/evaluation/eval_3DPW.py#L185
    """
    min_p1 = np.min(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p1 = np.max(p1, axis=0)
    max_p2 = np.max(p2, axis=0)

    bb1 = {}
    bb2 = {}

    bb1['x1'] = min_p1[0]
    bb1['x2'] = max_p1[0]
    bb1['y1'] = min_p1[1]
    bb1['y2'] = max_p1[1]
    bb2['x1'] = min_p2[0]
    bb2['x2'] = max_p2[0]
    bb2['y1'] = min_p2[1]
    bb2['y2'] = max_p2[1]

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou

    
def avg_per_human(lst, default=0.0):
    values = [x for arr in lst for x in np.array(arr).flatten()]
    return np.mean(values) if values else default

def extract_metrics(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    matches = re.findall(
        r'n_human: (\d+).*?PVE: ([\d.]+), PA-PVE: ([\d.]+), Metric-PVE: ([\d.]+), MPJPE: ([\d.]+), PA-MPJPE: ([\d.]+), Metric-MPJPE: ([\d.]+), RootError: ([\d.]+), W-MPJPE: ([\d.]+), WA-MPJPE: ([\d.]+), RTE: ([\d.]+), Scaled-RTE: ([\d.]+), Jitter: ([\d.]+), Foot-Sliding: ([\d.]+), Self-Pen-Depth: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-Score: ([\d.]+)',
        content
    )

    if not matches:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    total_humans = 0
    weighted_sums = [0.0] * 17

    for match in matches:
        n_human = int(match[0])
        metrics = [float(x) for x in match[1:]]

        total_humans += n_human
        for i, metric in enumerate(metrics):
            weighted_sums[i] += metric * n_human

    if total_humans == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    averages = [ws / total_humans for ws in weighted_sums]

    return total_humans, *averages


def process_directory(directory):
    results = []
    for root, _, files in os.walk(directory):
        for file in sorted(files):
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                result = extract_metrics(file_path)
                if result[0] > 0:
                    results.append(result)
    return results


def calculate_averages(results):
    if not results:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    total_humans = sum(r[0] for r in results)
    if total_humans == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    weighted_sums = [sum(r[i] * r[0] for r in results) for i in range(1, 18)]
    averages = [ws / total_humans for ws in weighted_sums]

    return total_humans, *averages

def visualize(
    save_dir, img_path, view, gt_v3d_c, pred_v3d_c, 
    K_to_proj, gt_K, bestMatch, smpl_face
):
    from dust3r.utils import render_meshes, denormalize_rgb
    from viser_utils import get_color
    from PIL import Image
    
    os.makedirs(save_dir, exist_ok=True)
    n_humans_i = pred_v3d_c.shape[0]

    # image
    img_array = denormalize_rgb(view.cpu().numpy())

    focal = K_to_proj[[0,1],[0,1]].cpu().numpy()
    princpt = K_to_proj[[0,1],[-1,-1]].cpu().numpy()
    gt_focal = gt_K[[0,1],[0,1]].cpu().numpy()
    gt_princpt = gt_K[[0,1],[-1,-1]].cpu().numpy()

    gt_color_indices = list(range(gt_v3d_c.shape[0]))
    pred_color_indices = [-1] * n_humans_i
    for (pid, gid) in bestMatch:
        pred_color_indices[pid] = gt_color_indices[gid]

    next_color_idx = max(gt_color_indices) + 1 if gt_color_indices else 0
    for i in range(len(pred_color_indices)):
        if pred_color_indices[i] == -1:
            pred_color_indices[i] = next_color_idx
            next_color_idx += 1

    # gt
    gt_verts, gt_faces = [], []
    for j in range(gt_v3d_c.shape[0]):
        gt_verts.append(gt_v3d_c[j].cpu().numpy().reshape(-1,3))
        gt_faces.append(smpl_face)

    gt_colors = [get_color(gt_color_indices[j])/255 for j in range(len(gt_verts))]
    gt_rend_array = render_meshes(img_array.copy(), 
                                    gt_verts, 
                                    gt_faces,
                                    {'focal': gt_focal, 'princpt': gt_princpt},
                                    color=gt_colors)
    
    # pred
    pred_verts, pred_faces = [], []
    for j in range(n_humans_i):
        pred_verts.append(pred_v3d_c[j].cpu().numpy().reshape(-1,3))
        pred_faces.append(smpl_face)
    
    pred_colors = [get_color(pred_color_indices[j])/255 for j in range(len(pred_verts))]
    pred_rend_array = render_meshes(img_array.copy(), 
                                    pred_verts, 
                                    pred_faces,
                                    {'focal': focal, 'princpt': princpt},
                                    color=pred_colors)

    img = np.concatenate([img_array, pred_rend_array, gt_rend_array], 1)
    Image.fromarray(img).save(
        os.path.join(
            f"{save_dir}/{os.path.splitext(os.path.basename(img_path))[0]}.jpg"
            ))

def write_log(log_path, dataset, seq, counter, metrics):
    with open(log_path, "a") as f:
        f.write(
            f"{dataset}-{seq: <16} |  "
            f"n_human: {counter['n_human']:06d} | "
            f"PVE: {avg_per_human(metrics['ca_pve']):.1f}, "
            f"PA-PVE: {avg_per_human(metrics['pa_pve']):.1f}, "
            f"Metric-PVE: {avg_per_human(metrics['me_pve']):.1f}, "
            f"MPJPE: {avg_per_human(metrics['ca_mpjpe']):.1f}, "
            f"PA-MPJPE: {avg_per_human(metrics['pa_mpjpe']):.1f}, "
            f"Metric-MPJPE: {avg_per_human(metrics['me_mpjpe']):.1f}, "
            f"RootError: {avg_per_human(metrics['rt_error']):.1f}, "
            f"W-MPJPE: {avg_per_human(metrics['wa2_mpjpe']):.1f}, "
            f"WA-MPJPE: {avg_per_human(metrics['waa_mpjpe']):.1f}, "
            f"RTE: {avg_per_human(metrics['rte']):.1f}, "
            f"Scaled-RTE: {avg_per_human(metrics['rte_scaled']):.1f}, "
            f"Jitter: {avg_per_human(metrics['jitter']):.1f}, "
            f"Foot-Sliding: {avg_per_human(metrics['fs']):.1f}, "
            f"Self-Pen-Depth: {avg_per_human(metrics.get('self_pen_depth', [])):.2f}, "
            f"Precision: {metrics['precision']:.1f}, "
            f"Recall: {metrics['recall']:.1f}, "
            f"F1-Score: {metrics['f1_score']:.1f}\n"
        )

def get_summary_log(summary):
    """Generate summary log for evaluation results"""
    return (
        f"EVALUATION SUMMARY\n"
        f"{'='*7}EVALUATION SUMMARY{'='*7}\n"
        f"Total Humans: {summary[0]}\n"
        f"\n"
        f"Camera Coordinate Metrics (mm):\n"
        f"  PVE:         {summary[1]:6.1f}\n"
        f"  PA-PVE:      {summary[2]:6.1f}\n"
        f"  Metric-PVE:  {summary[3]:6.1f}\n"
        f"\n"
        f"  MPJPE:       {summary[4]:6.1f}\n"
        f"  PA-MPJPE:    {summary[5]:6.1f}\n"
        f"  Metric-MPJPE: {summary[6]:6.1f}\n"
        f"  Root-Error:  {summary[7]:6.1f}\n"
        f"\n"
        f"Global Coordinate Metrics (cm):\n"
        f"  W-MPJPE:     {summary[8]:6.1f}\n"
        f"  WA-MPJPE:    {summary[9]:6.1f}\n"
        f"  RTE:         {summary[10]:6.1f}\n"
        f"  Scaled-RTE:  {summary[11]:6.1f}\n"
        f"  Jitter:      {summary[12]:6.1f}\n"
        f"  Foot-Sliding: {summary[13]:6.1f}\n"
        f"\n"
        f"Self-Penetration Metrics (VolumetricSMPL):\n"
        f"  Self-Pen-Depth (mm): {summary[14]:6.2f}\n"
        f"\n"
        f"Detection Metrics (%):\n"
        f"  Precision:   {summary[15]:6.1f}\n"
        f"  Recall:      {summary[16]:6.1f}\n"
        f"  F1-Score:    {summary[17]:6.1f}\n"
        f"{'='*32}\n"
    )
    

# Evaluation metrics
# Code modifed from https://github.com/zju3dv/GVHMR/blob/088caff492aa38c2d82cea363b78a3c65a83118f/hmr4d/utils/eval/eval_utils.py

def compute_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).numpy()

def compute_perjoint_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).numpy()

def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    S1 = S1.permute(0,2,1)
    S2 = S2.permute(0,2,1)
    transposed = True

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat

def batch_align_by_pelvis(data_list, pelvis_idxs):
    """
    Assumes data is given as [pred_j3d, target_j3d, pred_verts, target_verts].
    Each data is in shape of (batch, num_points, 3)
    Pelvis is notated as one / two joints indices.
    Align all data to the corresponding pelvis location.
    """

    pred_j3d, target_j3d, pred_verts, target_verts = data_list
    
    pred_pelvis = pred_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()
    target_pelvis = target_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()
    
    # Align to the pelvis
    pred_j3d = pred_j3d - pred_pelvis
    target_j3d = target_j3d - target_pelvis
    pred_verts = pred_verts - pred_pelvis
    target_verts = target_verts - target_pelvis
    
    return (pred_j3d, target_j3d, pred_verts, target_verts, pred_pelvis, target_pelvis)

def align_pcl(Y, X, weight=None, fixed_scale=False):
    """
    align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / var[..., 0]  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t

def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3))
    pred_glob = s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    return pred_glob

def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3))
    pred_first = s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    return pred_first

def compute_rte(target_trans, pred_trans, fixed_scale=True):
    # Compute the global alignment
    scale, rot, trans = align_pcl(target_trans[None, :], pred_trans[None, :], fixed_scale=fixed_scale)
    pred_trans_hat = (scale * torch.einsum("tij,tnj->tni", rot, pred_trans[None, :]) + trans[None, :])[0]

    # Compute the entire displacement of ground truth trajectory
    disps, disp = [], 0
    for p1, p2 in zip(target_trans, target_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)

    # Compute absolute root-translation-error (RTE)
    rte = torch.norm(target_trans - pred_trans_hat, 2, dim=-1)

    # Normalize it to the displacement
    return (rte / disp).numpy()


def compute_jitter(joints, fps=30):
    """compute jitter of the motion
    Args:
        joints (N, J, 3).
        fps (float).
    Returns:
        jitter (N-3).
    """
    pred_jitter = torch.norm(
        (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
        dim=2,
    ).mean(dim=-1)

    return pred_jitter.cpu().numpy() / 10.0


def compute_foot_sliding(target_verts, pred_verts, thr=1e-2):
    """compute foot sliding error
    The foot ground contact label is computed by the threshold of 1 cm/frame
    Args:
        target_verts (N, 6890, 3).
        pred_verts (N, 6890, 3).
    Returns:
        error (N frames in contact).
    """
    assert target_verts.shape == pred_verts.shape
    assert target_verts.shape[-2] == 6890

    # Foot vertices idxs
    foot_idxs = [3216, 3387, 6617, 6787]

    # Compute contact label
    foot_loc = target_verts[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr

    pred_feet_loc = pred_verts[:, foot_idxs]
    pred_disp = (pred_feet_loc[1:] - pred_feet_loc[:-1]).norm(2, dim=-1)

    error = pred_disp[contact]

    return error.cpu().numpy()

def compute_self_penetration(bm_x, rotvec, shape, expression=None):
    """Self-collision loss and mean penetration depth via VolumetricSMPL.

    bm_x       : smplx model with volume attached (smpl_layer.bm_x)
    rotvec     : (N, 53, 3) Human3R rotation vectors
    shape      : (N, num_betas) shape parameters
    expression : (N, 10) expression parameters or None

    Returns: mean_depth_mm float
      Mean SDF penetration depth (mm) of vertices with SDF < 0. Best used as
      a relative comparison metric — noise floor is consistent across methods.
    """
    n = rotvec.shape[0]
    dev = rotvec.device
    bm_x = bm_x.to(dev)

    with torch.no_grad():
        smpl_output = bm_x(
            global_orient=rotvec[:, 0].reshape(n, 3),
            body_pose=rotvec[:, 1:22].reshape(n, 63),
            left_hand_pose=rotvec[:, 22:37].reshape(n, 45),
            right_hand_pose=rotvec[:, 37:52].reshape(n, 45),
            jaw_pose=rotvec[:, 52].reshape(n, 3),
            betas=shape,
            expression=expression if expression is not None else bm_x.expression.to(dev).repeat(n, 1),
            leye_pose=bm_x.leye_pose.to(dev).repeat(n, 1),
            reye_pose=bm_x.reye_pose.to(dev).repeat(n, 1),
            return_full_pose=True,
        )
        sdf = bm_x.volume.query(smpl_output.vertices, smpl_output)  # (N, V)

    inside = sdf < 0
    mean_depth_mm = (-sdf[inside]).mean().item() * 1000.0 if inside.any() else 0.0
    return mean_depth_mm


def eval_camcoord(batch, pelvis_idxs=[1, 2], fps=30):
    """
    Args:
        batch (dict): {
            "pred_j3d": (..., J, 3) tensor
            "target_j3d":
            "pred_v3d":
            "target_v3d":
        }
    Returns:
        cam_coord_metrics (dict): {
            "pa_mpjpe": (..., ) numpy array
            "mpjpe":
            "pve":
            "accel":
        }
    """
    # All data is in camera coordinates
    pred_j3d = batch["pred_j3d"]  # (..., J, 3)
    target_j3d = batch["target_j3d"]
    pred_verts = batch["pred_v3d"]
    target_verts = batch["target_v3d"]

    # Center Align by pelvis
    (   ca_pred_j3d, ca_target_j3d, ca_pred_verts, ca_target_verts, pred_pelvis, target_pelvis
     ) = batch_align_by_pelvis(
        [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs
    )

    pa_pred_j3d = batch_compute_similarity_transform_torch(ca_pred_j3d, ca_target_j3d)
    pa_pred_verts = batch_compute_similarity_transform_torch(ca_pred_verts, ca_target_verts)

    # Metrics
    m2mm = 1000

    # metric scale
    rt_error = compute_jpe(pred_pelvis, target_pelvis) * m2mm
    me_mpjpe = compute_jpe(pred_j3d, target_j3d) * m2mm
    me_pve = compute_jpe(pred_verts, target_verts) * m2mm

    # center aligned
    ca_mpjpe = compute_jpe(ca_pred_j3d, ca_target_j3d) * m2mm
    ca_pve = compute_jpe(ca_pred_verts, ca_target_verts) * m2mm
   
    # procrustes aligned
    pa_mpjpe = compute_jpe(pa_pred_j3d, ca_target_j3d) * m2mm
    pa_pve = compute_jpe(pa_pred_verts, ca_target_verts) * m2mm
    
    camcoord_metrics = {
        "me_mpjpe": me_mpjpe,
        "ca_mpjpe": ca_mpjpe,
        "pa_mpjpe": pa_mpjpe,
        "me_pve": me_pve,
        "ca_pve": ca_pve,
        "pa_pve": pa_pve,
        "rt_error": rt_error,
    }
    return camcoord_metrics

def eval_global(batch, subsample=1):
    """Follow WHAM, the input has skipped invalid frames
    Args:
        batch (dict): {
            "pred_j3d": (F, J, 3) tensor
            "target_j3d":
            "pred_v3d":
            "target_v3d":
        }
    Returns:
        global_metrics (dict): {
            "wa2_mpjpe": (F, ) numpy array
            "waa_mpjpe":
            "rte":
            "jitter":
            "fs":
        }
    """
    # All data is in global coordinates
    pred_j3d_glob = batch["pred_j3d"]  # (..., J, 3)
    target_j3d_glob = batch["target_j3d"]
    pred_verts_glob = batch["pred_v3d"]
    target_verts_glob = batch["target_v3d"]

    seq_length = pred_j3d_glob.shape[0]

    # Use chunk to compare
    chunk_length = int(100 / subsample)
    wa2_mpjpe, waa_mpjpe = [], []
    for start in range(0, seq_length, chunk_length):
        end = min(seq_length, start + chunk_length)

        target_j3d = target_j3d_glob[start:end].clone().cpu()
        pred_j3d = pred_j3d_glob[start:end].clone().cpu()

        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)

        wa2_mpjpe.append(compute_jpe(target_j3d, w_j3d))
        waa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))

    # Metrics
    m2mm = 1000
    wa2_mpjpe = np.concatenate(wa2_mpjpe) * m2mm
    waa_mpjpe = np.concatenate(waa_mpjpe) * m2mm

    # Additional Metrics
    rte = compute_rte(target_j3d_glob[:, 0].cpu(), pred_j3d_glob[:, 0].cpu()) * 1e2
    rte_scaled = compute_rte(
        target_j3d_glob[:, 0].cpu(), pred_j3d_glob[:, 0].cpu(), fixed_scale=False) * 1e2
    jitter = compute_jitter(pred_j3d_glob, fps=30)
    foot_sliding = compute_foot_sliding(target_verts_glob, pred_verts_glob) * m2mm

    global_metrics = {
        "wa2_mpjpe": wa2_mpjpe,
        "waa_mpjpe": waa_mpjpe,
        "rte": rte,
        "rte_scaled": rte_scaled,
        "jitter": jitter,
        "fs": foot_sliding,
    }
    return global_metrics