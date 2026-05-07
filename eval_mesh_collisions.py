#!/usr/bin/env python3
"""
Human-human and self-collision mesh metrics for Human3R scene.pkl outputs.

This script expects the bundle saved by:
    python demo.py ... --save_scene

It evaluates:
  1. Human-human penetration:
     vertices of person A that are inside person B's SMPL-X mesh, and vice versa.
  2. Self-collision proxy:
     non-local vertices on the same SMPL-X mesh that are closer than a threshold.

The human-human metric uses a solid-angle winding number for inside/outside tests
and an approximate point-to-triangle distance for penetration depth. The self
metric is intentionally named a proxy: exact SMPL-X self-intersection is best
done with body-part segmentation plus a BVH/SDF backend, but this catches many
bad arm-through-body / leg-through-leg cases without extra dependencies.
"""

import argparse
import csv
import glob
import os
import pickle
import sys
from collections import deque

import numpy as np
import torch

COARSE_SMPLX_PARTS = {
    "pelvis": "pelvis",
    "left_hip": "left_thigh",
    "right_hip": "right_thigh",
    "spine1": "torso",
    "left_knee": "left_shin",
    "right_knee": "right_shin",
    "spine2": "torso",
    "left_ankle": "left_foot",
    "right_ankle": "right_foot",
    "spine3": "torso",
    "left_foot": "left_foot",
    "right_foot": "right_foot",
    "neck": "neck",
    "left_collar": "left_shoulder",
    "right_collar": "right_shoulder",
    "head": "head",
    "left_shoulder": "left_upper_arm",
    "right_shoulder": "right_upper_arm",
    "left_elbow": "left_forearm",
    "right_elbow": "right_forearm",
    "left_wrist": "left_hand",
    "right_wrist": "right_hand",
    "jaw": "head",
    "left_eye_smplhf": "head",
    "right_eye_smplhf": "head",
}

for _side in ("left", "right"):
    for _finger in ("index", "middle", "pinky", "ring", "thumb"):
        for _idx in ("1", "2", "3"):
            COARSE_SMPLX_PARTS[f"{_side}_{_finger}{_idx}"] = f"{_side}_hand"


def parse_args():
    # The evaluator can consume either:
    #   1. scene.pkl from demo.py --save_scene, where vertices are already saved.
    #   2. an output directory from demo.py --save, where SMPL-X params must be
    #      converted back to vertices through Human3R's SMPL_Layer.
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="Path to scene.pkl saved by demo.py --save_scene.")
    p.add_argument("--output_dir", default=None,
                   help="Directory saved by demo.py --save, containing smpl/*.npz and camera/*.npz.")
    p.add_argument("--frame", type=int, default=None, help="Evaluate only one frame index.")
    p.add_argument("--hh_tolerance_mm", type=float, default=5.0,
                   help="Inside vertices with depth below this are ignored.")
    p.add_argument("--self_tolerance_mm", type=float, default=20.0,
                   help="Non-local same-body vertices closer than this are self-collision candidates.")
    p.add_argument("--unit_scale_to_m", type=float, default=1.0,
                   help="Multiplier from scene units to meters. Human3R is approximately metric.")
    p.add_argument("--max_query_points", type=int, default=1500,
                   help="Max vertices sampled per human for each test. Use 0 for all vertices.")
    p.add_argument("--face_stride", type=int, default=1,
                   help="Use every Nth face for speed. 1 is most accurate.")
    p.add_argument("--self_adjacency_rings", type=int, default=2,
                   help="Ignore vertices within this many mesh-edge hops for the self proxy.")
    p.add_argument("--save_csv", default=None, help="Optional CSV path for per-frame metrics.")
    p.add_argument("--save_part_csv", default=None,
                   help="Optional CSV path for part-level collision rows.")
    return p.parse_args()


def as_frame_verts(raw):
    # Normalize all frame vertex containers to:
    #   (num_humans, num_vertices, 3)
    # Empty frames become (0, 0, 3), and a single mesh (V, 3) becomes (1, V, 3).
    verts = np.asarray(raw, dtype=np.float32)
    if verts.size == 0:
        return np.empty((0, 0, 3), dtype=np.float32)
    if verts.ndim == 2:
        verts = verts[None]
    return verts


def ensure_repo_imports():
    # When running the script directly from the repo root, make src/ importable
    # so "from dust3r..." works without installing Human3R as a package.
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def npz_value(data, key):
    # np.savez stores None values as zero-dimensional object arrays. Convert
    # those back into plain Python objects before downstream shape checks.
    value = data[key]
    if value.shape == () and value.dtype == object:
        return value.item()
    return value


def build_part_labels_from_smpl_layer(layer):
    """Assign each SMPL-X vertex to a coarse body part using max LBS weight."""
    # SMPL-X does not ship a part CSV in this repo. As a useful fallback, each
    # vertex is assigned to the joint that influences it most strongly in the
    # linear-blend-skinning weights, then joints are collapsed into coarse parts
    # such as torso, left_hand, right_thigh, head, etc.
    weights = layer.bm_x.lbs_weights.detach().cpu().numpy()
    joint_ids = weights.argmax(axis=1)
    joint_names = getattr(layer, "joint_names", [])
    part_names = []
    vertex_part_ids = np.zeros(len(joint_ids), dtype=np.int64)
    part_to_id = {}

    for v_id, joint_id in enumerate(joint_ids):
        if int(joint_id) < len(joint_names):
            joint_name = joint_names[int(joint_id)]
        else:
            joint_name = f"joint_{int(joint_id)}"
        part_name = COARSE_SMPLX_PARTS.get(joint_name, joint_name)
        if part_name not in part_to_id:
            part_to_id[part_name] = len(part_names)
            part_names.append(part_name)
        vertex_part_ids[v_id] = part_to_id[part_name]

    return np.asarray(part_names, dtype=object), vertex_part_ids


def face_part_ids_from_vertices(faces, vertex_part_ids):
    """Face label is the majority part among its three vertices."""
    # Human-human target parts are measured on the nearest target triangle.
    # To name that target triangle, give each face the majority label of its
    # three vertices.
    out = np.zeros(len(faces), dtype=np.int64)
    for f_id, face in enumerate(faces):
        labels = vertex_part_ids[face]
        vals, counts = np.unique(labels, return_counts=True)
        out[f_id] = vals[counts.argmax()]
    return out


def load_scene_bundle(scene_path):
    # scene.pkl already contains world/camera-space mesh vertices and faces.
    # It usually does not contain part labels, so part-level reporting is only
    # available automatically for --output_dir unless labels are added there.
    with open(scene_path, "rb") as f:
        scene = pickle.load(f)
    verts = [as_frame_verts(v) for v in scene["verts"]]
    faces = np.asarray(scene["smpl_faces"], dtype=np.int64)
    return verts, faces, None, None


def load_saved_output_dir(output_dir, frame=None):
    """Reconstruct per-frame SMPL-X vertices from demo.py --save outputs."""
    output_dir = output_dir.strip()
    ensure_repo_imports()
    try:
        from dust3r.utils.smpl_layer import SMPL_Layer
    except Exception as exc:
        raise RuntimeError(
            "Could not import Human3R SMPL_Layer. Run this from the Human3R repo "
            "with the same environment you used for demo.py, where smplx and roma "
            "are installed."
        ) from exc

    smpl_files = sorted(glob.glob(os.path.join(output_dir, "smpl", "*.npz")))
    if not smpl_files:
        raise FileNotFoundError(f"No smpl/*.npz files found under {output_dir}")
    if frame is not None:
        wanted = os.path.join(output_dir, "smpl", f"{frame:06d}.npz")
        if not os.path.exists(wanted):
            raise FileNotFoundError(f"Requested frame {frame}, but {wanted} does not exist.")
        smpl_files = [wanted]

    layers = {}
    verts_by_frame = []
    faces = None
    part_names = None
    vertex_part_ids = None

    for file_i, smpl_path in enumerate(smpl_files):
        # Full-sequence evaluation may read hundreds of frames from a shared
        # filesystem. Print coarse progress so long runs do not look frozen.
        if frame is None and (file_i == 0 or (file_i + 1) % 25 == 0 or file_i + 1 == len(smpl_files)):
            print(f"Reconstructing SMPL-X meshes: {file_i + 1}/{len(smpl_files)}", flush=True)
        frame_name = os.path.splitext(os.path.basename(smpl_path))[0]
        cam_path = os.path.join(output_dir, "camera", f"{frame_name}.npz")
        smpl_data = np.load(smpl_path, allow_pickle=True)

        shape = np.asarray(npz_value(smpl_data, "shape"), dtype=np.float32)
        num_betas = int(shape.shape[-1]) if shape.ndim >= 2 else 10
        if num_betas not in layers:
            # Cache one SMPL layer per beta dimensionality. Human3R saved runs
            # may use 10 or 11 betas depending on the source/config.
            layers[num_betas] = SMPL_Layer(
                type="smplx",
                gender="neutral",
                num_betas=num_betas,
                kid=False,
                person_center="head",
            )
            if faces is None:
                faces = np.asarray(layers[num_betas].bm_x.faces, dtype=np.int64)
                part_names, vertex_part_ids = build_part_labels_from_smpl_layer(layers[num_betas])

        if shape.size == 0:
            # This frame has no detected humans. Keep a placeholder frame so
            # frame indices in the output still match the original sequence.
            verts_by_frame.append(np.empty((0, 0, 3), dtype=np.float32))
            continue

        rotvec = np.asarray(npz_value(smpl_data, "rotvec"), dtype=np.float32)
        transl = np.asarray(npz_value(smpl_data, "transl"), dtype=np.float32)
        expression = npz_value(smpl_data, "expression")
        expression = None if expression is None else np.asarray(expression, dtype=np.float32)

        if os.path.exists(cam_path):
            cam_data = np.load(cam_path, allow_pickle=True)
            K = np.asarray(cam_data["intrinsics"], dtype=np.float32)
        else:
            K = np.eye(3, dtype=np.float32)

        layer = layers[num_betas]
        n_humans = int(shape.shape[0])
        with torch.no_grad():
            # Convert saved SMPL-X parameters back to actual mesh vertices.
            # SMPL_Layer returns camera-space vertices in out["smpl_v3d"].
            out = layer(
                torch.from_numpy(rotvec).float(),
                torch.from_numpy(shape).float(),
                torch.from_numpy(transl).float(),
                None,
                None,
                K=torch.from_numpy(K).float().expand(n_humans, -1, -1),
                expression=None if expression is None else torch.from_numpy(expression).float(),
            )
        verts_by_frame.append(out["smpl_v3d"].cpu().numpy().astype(np.float32))

    if faces is None:
        raise RuntimeError("Could not determine SMPL-X faces because every frame has zero humans.")
    return verts_by_frame, faces, part_names, vertex_part_ids


def sample_indices(n, max_points):
    # For speed, optionally evaluate a deterministic uniform subset of vertices.
    # max_points <= 0 means "use all vertices".
    if max_points is None or max_points <= 0 or n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).round().astype(np.int64)


def aabb_overlap(a, b, margin):
    # Cheap broad-phase collision rejection. If bounding boxes do not overlap,
    # the expensive winding-number inside test cannot find penetration.
    amin, amax = a.min(axis=0) - margin, a.max(axis=0) + margin
    bmin, bmax = b.min(axis=0) - margin, b.max(axis=0) + margin
    return bool(np.all(amax >= bmin) and np.all(bmax >= amin))


def winding_number(points, verts, faces, point_chunk=256):
    """Generalized winding number. abs(value) > 0.5 means inside a closed mesh."""
    # For a closed triangle mesh, the solid angle around an outside point sums
    # to ~0, while the solid angle around an inside point sums to +/-4*pi.
    # Dividing by 4*pi yields a winding number near 0 outside and +/-1 inside.
    tris = verts[faces].astype(np.float64)
    out = np.zeros(len(points), dtype=np.float64)
    eps = 1e-12

    for start in range(0, len(points), point_chunk):
        p = points[start:start + point_chunk].astype(np.float64)
        total = np.zeros(len(p), dtype=np.float64)

        # Keep this loop explicit to avoid allocating a huge P x F x 3 tensor.
        for tri in tris:
            a = tri[0] - p
            b = tri[1] - p
            c = tri[2] - p
            la = np.linalg.norm(a, axis=1)
            lb = np.linalg.norm(b, axis=1)
            lc = np.linalg.norm(c, axis=1)
            numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
            denominator = (
                la * lb * lc
                + np.einsum("ij,ij->i", a, b) * lc
                + np.einsum("ij,ij->i", b, c) * la
                + np.einsum("ij,ij->i", c, a) * lb
            )
            total += 2.0 * np.arctan2(numerator, denominator + eps)

        out[start:start + point_chunk] = total / (4.0 * np.pi)
    return out


def point_triangle_min_distance(points, verts, faces, face_chunk=2048, device="cpu", return_face_ids=False):
    """Approximate point-to-mesh distance with point-to-triangle primitives."""
    # For penetrating vertices, this estimates how deep they are by finding the
    # nearest triangle on the target mesh. It also optionally returns the target
    # triangle id, which is used to name the target body part.
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    v = torch.as_tensor(verts, dtype=torch.float32, device=device)
    f = torch.as_tensor(faces, dtype=torch.long, device=device)
    best = torch.full((pts.shape[0],), float("inf"), dtype=torch.float32, device=device)
    best_face = torch.full((pts.shape[0],), -1, dtype=torch.long, device=device)
    eps = 1e-12

    def edge_dist2(p, a, b):
        # Squared distance to a triangle edge segment. Used when the projected
        # point falls outside the triangle interior.
        ab = b - a
        t = ((p - a) * ab).sum(-1) / (ab.square().sum(-1) + eps)
        t = t.clamp(0.0, 1.0)
        q = a + t[..., None] * ab
        return (p - q).square().sum(-1)

    p = pts[:, None, :]
    for start in range(0, f.shape[0], face_chunk):
        tri = v[f[start:start + face_chunk]]
        a = tri[None, :, 0]
        b = tri[None, :, 1]
        c = tri[None, :, 2]

        ab = b - a
        ac = c - a
        n = torch.cross(ab, ac, dim=-1)
        n2 = n.square().sum(-1).clamp_min(eps)
        signed = ((p - a) * n).sum(-1) / torch.sqrt(n2)
        proj = p - signed[..., None] * n / n2[..., None]

        v0 = ab
        v1 = ac
        v2 = proj - a
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0).sum(-1)
        d21 = (v2 * v1).sum(-1)
        denom = (d00 * d11 - d01 * d01).clamp_min(eps)
        bary_v = (d11 * d20 - d01 * d21) / denom
        bary_w = (d00 * d21 - d01 * d20) / denom
        bary_u = 1.0 - bary_v - bary_w
        inside = (bary_u >= 0) & (bary_v >= 0) & (bary_w >= 0)
        # If the orthogonal projection lies inside the triangle, the point-to-
        # plane distance is valid. Otherwise the closest point must lie on an
        # edge, handled below.
        plane_d2 = torch.where(inside, signed.square(), torch.full_like(signed, float("inf")))

        d2 = torch.minimum(plane_d2, edge_dist2(p, a, b))
        d2 = torch.minimum(d2, edge_dist2(p, b, c))
        d2 = torch.minimum(d2, edge_dist2(p, c, a))
        chunk_min, chunk_arg = d2.min(dim=1)
        better = chunk_min < best
        best = torch.where(better, chunk_min, best)
        best_face = torch.where(better, chunk_arg + start, best_face)

    dist = torch.sqrt(best).cpu().numpy()
    if return_face_ids:
        return dist, best_face.cpu().numpy()
    return dist


def build_vertex_adjacency(num_verts, faces):
    # Build mesh-neighborhood graph from faces. This is used by the self proxy
    # to ignore immediately adjacent vertices that are naturally close.
    adj = [set() for _ in range(num_verts)]
    for i, j, k in faces:
        adj[i].update((j, k))
        adj[j].update((i, k))
        adj[k].update((i, j))
    return adj


def expand_neighbors(adj, root, rings):
    # Return vertices within N edge hops of root. These vertices are excluded
    # from the self-proximity check to avoid counting normal local mesh density.
    if root >= len(adj):
        return {root}
    seen = {root}
    q = deque([(root, 0)])
    while q:
        cur, depth = q.popleft()
        if depth >= rings:
            continue
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, depth + 1))
    return seen


def add_part_row(rows, metric_type, frame_id, src_person, src_part, dst_person, dst_part, depths_m, points):
    # Accumulate one row for a body-part pair. The center is the average 3D
    # position of the violating source points, useful for visualization/debugging.
    depths_m = np.asarray(depths_m, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if len(depths_m) == 0:
        return
    center = points.mean(axis=0) if len(points) else np.zeros(3, dtype=np.float64)
    rows.append({
        "frame": frame_id,
        "metric_type": metric_type,
        "violated": True,
        "src_person": int(src_person),
        "src_part": str(src_part),
        "dst_person": int(dst_person),
        "dst_part": str(dst_part),
        "n_points": int(len(depths_m)),
        "mean_depth_mm": float(depths_m.mean() * 1000.0),
        "max_depth_mm": float(depths_m.max() * 1000.0),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
    })


def self_collision_proxy(
    verts, faces, adj, tolerance_m, max_points, adjacency_rings,
    part_names=None, vertex_part_ids=None, frame_id=None, human_id=None,
):
    # Self collision here is a proxy, not exact triangle-triangle intersection:
    # for sampled vertices, find the nearest non-local vertex on the same body.
    # If it is closer than tolerance_m, count a violation.
    if len(verts) == 0:
        return (
            {"n_query": 0, "n_close": 0, "pct_close": 0.0, "mean_violation_mm": 0.0,
             "min_nonlocal_dist_mm": 0.0},
            [],
        )

    query_ids = sample_indices(len(verts), max_points)
    all_v = torch.as_tensor(verts, dtype=torch.float32)
    close = []
    min_dists = []
    part_rows_raw = {}

    for idx in query_ids:
        local = expand_neighbors(adj, int(idx), rings=adjacency_rings)
        p = torch.as_tensor(verts[idx], dtype=torch.float32)[None]
        d = torch.cdist(p, all_v)[0]
        if local:
            d[torch.as_tensor(sorted(local), dtype=torch.long)] = float("inf")
        nearest_id = int(d.argmin())
        md = float(d[nearest_id])
        min_dists.append(md)
        if md < tolerance_m:
            # A violation is "how much closer than the allowed threshold" the
            # nearest non-local vertex is.
            violation = tolerance_m - md
            close.append(violation)
            if part_names is not None and vertex_part_ids is not None:
                src_part = str(part_names[vertex_part_ids[int(idx)]])
                dst_part = str(part_names[vertex_part_ids[nearest_id]])
                key = (src_part, dst_part)
                if key not in part_rows_raw:
                    part_rows_raw[key] = {"depths": [], "points": []}
                part_rows_raw[key]["depths"].append(violation)
                part_rows_raw[key]["points"].append(verts[int(idx)])

    if close:
        mean_violation_mm = float(np.mean(close) * 1000.0)
    else:
        mean_violation_mm = 0.0

    metrics = {
        "n_query": int(len(query_ids)),
        "n_close": int(len(close)),
        "pct_close": float(len(close) / max(len(query_ids), 1)),
        "mean_violation_mm": mean_violation_mm,
        "min_nonlocal_dist_mm": float(np.min(min_dists) * 1000.0) if min_dists else 0.0,
    }
    part_rows = []
    if part_names is not None and vertex_part_ids is not None:
        for (src_part, dst_part), data in part_rows_raw.items():
            add_part_row(
                part_rows, "self", frame_id, human_id, src_part, human_id, dst_part,
                np.asarray(data["depths"]), np.asarray(data["points"]),
            )
    return metrics, part_rows


def human_human_pair_metrics(
    verts_a, verts_b, faces, tolerance_m, max_points,
    part_names=None, vertex_part_ids=None, face_part_ids=None,
    frame_id=None, person_a=None, person_b=None,
):
    # Evaluate both directions:
    #   vertices of A inside the closed mesh of B
    #   vertices of B inside the closed mesh of A
    # This catches asymmetric samples and lets us name the source body part.
    if not aabb_overlap(verts_a, verts_b, margin=tolerance_m):
        return (
            {"n_query": 0, "n_inside": 0, "pct_inside": 0.0,
             "mean_penetration_mm": 0.0, "max_penetration_mm": 0.0},
            [],
        )

    part_rows_raw = {}

    def one_direction(src, dst, src_person, dst_person):
        # Winding number tells us which source vertices are inside dst.
        # Point-to-triangle distance then approximates penetration depth.
        ids = sample_indices(len(src), max_points)
        pts = src[ids]
        w = winding_number(pts, dst, faces)
        inside = np.abs(w) > 0.5
        if not inside.any():
            return len(pts), np.empty((0,), dtype=np.float32)
        inside_pts = pts[inside]
        inside_ids = ids[inside]
        depths, nearest_faces = point_triangle_min_distance(
            inside_pts, dst, faces, return_face_ids=True,
        )
        keep = depths > tolerance_m
        depths = depths[keep]
        if part_names is not None and vertex_part_ids is not None and face_part_ids is not None:
            # Source part: label of the penetrating source vertex.
            # Target part: label of the nearest target triangle.
            kept_pts = inside_pts[keep]
            kept_ids = inside_ids[keep]
            kept_faces = nearest_faces[keep]
            for point, src_v_id, dst_face_id, depth in zip(kept_pts, kept_ids, kept_faces, depths):
                src_part = str(part_names[vertex_part_ids[int(src_v_id)]])
                dst_part = str(part_names[face_part_ids[int(dst_face_id)]])
                key = (src_person, src_part, dst_person, dst_part)
                if key not in part_rows_raw:
                    part_rows_raw[key] = {"depths": [], "points": []}
                part_rows_raw[key]["depths"].append(depth)
                part_rows_raw[key]["points"].append(point)
        return len(pts), depths

    n_ab, d_ab = one_direction(verts_a, verts_b, person_a, person_b)
    n_ba, d_ba = one_direction(verts_b, verts_a, person_b, person_a)
    depths = np.concatenate([d_ab, d_ba])
    n_query = n_ab + n_ba

    metrics = {
        "n_query": int(n_query),
        "n_inside": int(len(depths)),
        "pct_inside": float(len(depths) / max(n_query, 1)),
        "mean_penetration_mm": float(depths.mean() * 1000.0) if len(depths) else 0.0,
        "max_penetration_mm": float(depths.max() * 1000.0) if len(depths) else 0.0,
    }
    part_rows = []
    for (src_person, src_part, dst_person, dst_part), data in part_rows_raw.items():
        add_part_row(
            part_rows, "human_human", frame_id, src_person, src_part, dst_person, dst_part,
            np.asarray(data["depths"]), np.asarray(data["points"]),
        )
    return metrics, part_rows


def evaluate_frame(frame_id, verts, faces, adj, args, part_names=None, vertex_part_ids=None, face_part_ids=None):
    # Convert user thresholds from millimeters into the mesh unit. Human3R is
    # approximately metric, so by default 1 scene unit is treated as 1 meter.
    tolerance_hh = args.hh_tolerance_mm / 1000.0 / args.unit_scale_to_m
    tolerance_self = args.self_tolerance_mm / 1000.0 / args.unit_scale_to_m

    n_humans = int(len(verts))
    hh_pairs = []
    part_rows = []
    for i in range(n_humans):
        for j in range(i + 1, n_humans):
            # Human-human is evaluated for every unordered pair in the frame.
            m, rows = human_human_pair_metrics(
                verts[i], verts[j], faces, tolerance_hh, args.max_query_points,
                part_names=part_names,
                vertex_part_ids=vertex_part_ids,
                face_part_ids=face_part_ids,
                frame_id=frame_id,
                person_a=i,
                person_b=j,
            )
            m["pair"] = f"{i}-{j}"
            hh_pairs.append(m)
            part_rows.extend(rows)

    self_metrics = []
    for i in range(n_humans):
        # Self proxy is evaluated independently for each detected person.
        m, rows = self_collision_proxy(
            verts[i], faces, adj, tolerance_self, args.max_query_points,
            args.self_adjacency_rings,
            part_names=part_names,
            vertex_part_ids=vertex_part_ids,
            frame_id=frame_id,
            human_id=i,
        )
        m["human"] = i
        self_metrics.append(m)
        part_rows.extend(rows)

    hh_inside = sum(m["n_inside"] for m in hh_pairs)
    hh_query = sum(m["n_query"] for m in hh_pairs)
    hh_depth_sum = sum(m["mean_penetration_mm"] * m["n_inside"] for m in hh_pairs)
    hh_max = max([m["max_penetration_mm"] for m in hh_pairs] or [0.0])

    self_close = sum(m["n_close"] for m in self_metrics)
    self_query = sum(m["n_query"] for m in self_metrics)
    self_vio_sum = sum(m["mean_violation_mm"] * m["n_close"] for m in self_metrics)
    self_min = min([m["min_nonlocal_dist_mm"] for m in self_metrics if m["n_query"] > 0] or [0.0])

    return {
        "frame": frame_id,
        "n_humans": n_humans,
        "hh_pairs": len(hh_pairs),
        "hh_violated": bool(hh_inside > 0),
        "hh_n_inside": hh_inside,
        "hh_pct_inside": hh_inside / max(hh_query, 1),
        "hh_mean_penetration_mm": hh_depth_sum / max(hh_inside, 1),
        "hh_max_penetration_mm": hh_max,
        "self_violated": bool(self_close > 0),
        "self_n_close": self_close,
        "self_pct_close": self_close / max(self_query, 1),
        "self_mean_violation_mm": self_vio_sum / max(self_close, 1),
        "self_min_nonlocal_dist_mm": self_min,
        "any_violated": bool(hh_inside > 0 or self_close > 0),
    }, hh_pairs, self_metrics, part_rows


def print_result(summary, pair_rows, self_rows):
    print(f"\nFrame {summary['frame']} | humans: {summary['n_humans']}")
    print("  Human-human:")
    print(f"    pairs evaluated:        {summary['hh_pairs']}")
    print(f"    penetrating vertices:   {summary['hh_n_inside']}")
    print(f"    % sampled penetrating:  {summary['hh_pct_inside'] * 100.0:.2f}%")
    print(f"    mean depth:             {summary['hh_mean_penetration_mm']:.2f} mm")
    print(f"    max depth:              {summary['hh_max_penetration_mm']:.2f} mm")
    for row in pair_rows:
        if row["n_inside"] > 0:
            print(f"      pair {row['pair']}: {row['n_inside']} verts, "
                  f"mean {row['mean_penetration_mm']:.2f} mm, max {row['max_penetration_mm']:.2f} mm")

    print("  Self proxy:")
    print(f"    close non-local verts:  {summary['self_n_close']}")
    print(f"    % sampled close:        {summary['self_pct_close'] * 100.0:.2f}%")
    print(f"    mean violation:         {summary['self_mean_violation_mm']:.2f} mm")
    print(f"    min non-local distance: {summary['self_min_nonlocal_dist_mm']:.2f} mm")
    for row in self_rows:
        if row["n_close"] > 0:
            print(f"      human {row['human']}: {row['n_close']} verts, "
                  f"mean violation {row['mean_violation_mm']:.2f} mm")


def print_part_result(part_rows, max_rows=12):
    # Keep terminal output readable by printing only the strongest part-level
    # rows. The full list can still be written with --save_part_csv.
    if not part_rows:
        return
    print("  Part-level collisions:")
    rows = sorted(part_rows, key=lambda r: r["max_depth_mm"], reverse=True)
    for row in rows[:max_rows]:
        print(
            f"    {row['metric_type']}: person {row['src_person']} {row['src_part']} -> "
            f"person {row['dst_person']} {row['dst_part']} | "
            f"n={row['n_points']}, mean={row['mean_depth_mm']:.2f} mm, "
            f"max={row['max_depth_mm']:.2f} mm, "
            f"center=({row['center_x']:.3f}, {row['center_y']:.3f}, {row['center_z']:.3f})"
        )


def save_csv(rows, path):
    # Per-frame summary. Boolean *_violated columns make it easy to filter
    # frames with any human-human or self-proxy issue.
    fields = [
        "frame", "n_humans", "hh_pairs", "hh_violated", "hh_n_inside",
        "hh_pct_inside", "hh_mean_penetration_mm", "hh_max_penetration_mm",
        "self_violated", "self_n_close", "self_pct_close",
        "self_mean_violation_mm", "self_min_nonlocal_dist_mm", "any_violated",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def save_part_csv(rows, path):
    # Part-level rows. Every row represents a body-part pair that violated the
    # relevant threshold; "violated" is always True for these emitted rows.
    fields = [
        "frame", "metric_type", "violated", "src_person", "src_part",
        "dst_person", "dst_part", "n_points", "mean_depth_mm", "max_depth_mm",
        "center_x", "center_y", "center_z",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def main():
    args = parse_args()
    # Exactly one input source is required. Passing both is ambiguous, passing
    # neither gives the evaluator no mesh data.
    if (args.scene is None) == (args.output_dir is None):
        raise SystemExit("Pass exactly one of --scene or --output_dir.")

    if args.scene is not None:
        verts_by_frame, faces, part_names, vertex_part_ids = load_scene_bundle(args.scene)
    else:
        verts_by_frame, faces, part_names, vertex_part_ids = load_saved_output_dir(
            args.output_dir, frame=args.frame,
        )
    single_output_frame_loaded = args.output_dir is not None and args.frame is not None

    if args.face_stride > 1:
        # Optional narrow-phase speedup: fewer target triangles means faster
        # winding and distance checks, but less accurate penetration depths.
        faces = faces[::args.face_stride]
    face_part_ids = None
    if part_names is not None and vertex_part_ids is not None:
        face_part_ids = face_part_ids_from_vertices(faces, vertex_part_ids)
    elif args.save_part_csv:
        print("Part CSV requested, but no SMPL-X part labels are available for this input.")

    n_verts_from_faces = int(max(faces.reshape(-1)) + 1)
    n_verts_from_data = max(
        [int(v.shape[1]) for v in verts_by_frame if v.ndim == 3 and v.shape[0] > 0] or [0]
    )
    n_verts = max(n_verts_from_faces, n_verts_from_data)
    # Size adjacency using both faces and actual vertex arrays. This matters
    # when --face_stride drops faces that contain high-index vertices.
    adj = build_vertex_adjacency(n_verts, faces)

    frame_ids = [args.frame] if args.frame is not None else list(range(len(verts_by_frame)))
    summaries = []
    all_part_rows = []
    for local_idx, frame_id in enumerate(frame_ids):
        # With --output_dir --frame, only one frame is loaded, so local index 0
        # corresponds to the requested original frame id.
        verts_idx = local_idx if single_output_frame_loaded else frame_id
        verts = as_frame_verts(verts_by_frame[verts_idx])
        summary, pair_rows, self_rows, part_rows = evaluate_frame(
            frame_id, verts, faces, adj, args,
            part_names=part_names,
            vertex_part_ids=vertex_part_ids,
            face_part_ids=face_part_ids,
        )
        summaries.append(summary)
        all_part_rows.extend(part_rows)
        print_result(summary, pair_rows, self_rows)
        print_part_result(part_rows)

    if len(summaries) > 1:
        hh_inside = sum(r["hh_n_inside"] for r in summaries)
        self_close = sum(r["self_n_close"] for r in summaries)
        print("\nSequence summary")
        print(f"  frames evaluated:          {len(summaries)}")
        print(f"  total HH penetrating verts:{hh_inside}")
        print(f"  total self close verts:    {self_close}")
        print(f"  mean HH pct:               {np.mean([r['hh_pct_inside'] for r in summaries]) * 100.0:.2f}%")
        print(f"  mean self pct:             {np.mean([r['self_pct_close'] for r in summaries]) * 100.0:.2f}%")

    if args.save_csv:
        save_csv(summaries, args.save_csv)
        print(f"\nCSV saved to {args.save_csv}")

    if args.save_part_csv:
        save_part_csv(all_part_rows, args.save_part_csv)
        print(f"Part CSV saved to {args.save_part_csv}")


if __name__ == "__main__":
    main()
