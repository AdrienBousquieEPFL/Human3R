#!/usr/bin/env python3
"""Compare two demo.py --save output directories tensor-by-tensor.

Usage:
    python compare_runs.py tmp_before_refactor tmp_after_refactor

Walks both directories. For every .npy / .npz under the matching relative path,
loads the array(s) and reports any mismatch. PNGs are compared decoded to
numpy so PNG metadata noise doesn't cause false positives.
"""
import os
import sys
import numpy as np


def iter_rel_files(root):
    out = set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            out.add(os.path.relpath(os.path.join(dirpath, f), root))
    return out


def cmp_array(a, b):
    if a.shape != b.shape:
        return False, f"shape {a.shape} != {b.shape}"
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} != {b.dtype}"
    if np.array_equal(a, b):
        return True, "exact"
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    nan_mask = np.isnan(af) | np.isnan(bf)
    if nan_mask.all():
        return True, "all-NaN"
    diff = np.abs(af[~nan_mask] - bf[~nan_mask])
    max_diff = float(diff.max()) if diff.size else 0.0
    if max_diff == 0.0 and (np.isnan(af) == np.isnan(bf)).all():
        return True, "exact (NaN-aware)"
    if max_diff < 1e-6:
        return True, f"close max={max_diff:.2e}"
    return False, f"max={max_diff:.4e}"


def main(before_dir, after_dir):
    bf = iter_rel_files(before_dir)
    af = iter_rel_files(after_dir)
    only_b = sorted(bf - af)
    only_a = sorted(af - bf)
    if only_b:
        print(f"Only in BEFORE ({len(only_b)}): {only_b[:5]}{'...' if len(only_b) > 5 else ''}")
    if only_a:
        print(f"Only in AFTER  ({len(only_a)}): {only_a[:5]}{'...' if len(only_a) > 5 else ''}")

    common = sorted(bf & af)
    n_files = 0
    n_mismatch = 0
    failures = []

    for rel in common:
        bp = os.path.join(before_dir, rel)
        ap = os.path.join(after_dir, rel)
        try:
            if rel.endswith(".npy"):
                b = np.load(bp, allow_pickle=True)
                a = np.load(ap, allow_pickle=True)
                if b.dtype == object or a.dtype == object:
                    continue
                ok, msg = cmp_array(b, a)
                if not ok:
                    failures.append((rel, msg))
                    n_mismatch += 1
                n_files += 1
            elif rel.endswith(".npz"):
                b = np.load(bp, allow_pickle=True)
                a = np.load(ap, allow_pickle=True)
                if set(b.files) != set(a.files):
                    failures.append((rel, f"keys differ: {set(b.files) ^ set(a.files)}"))
                    n_mismatch += 1
                    continue
                npz_bad = []
                for k in b.files:
                    bk = b[k]
                    ak = a[k]
                    if bk.dtype == object or ak.dtype == object:
                        continue
                    ok, msg = cmp_array(bk, ak)
                    if not ok:
                        npz_bad.append(f"[{k}] {msg}")
                if npz_bad:
                    failures.append((rel, "; ".join(npz_bad)))
                    n_mismatch += 1
                n_files += 1
            elif rel.endswith(".png"):
                try:
                    import imageio.v2 as iio
                    b = iio.imread(bp)
                    a = iio.imread(ap)
                except Exception as e:
                    failures.append((rel, f"imread failed: {e}"))
                    n_mismatch += 1
                    continue
                ok, msg = cmp_array(b, a)
                if not ok:
                    failures.append((rel, msg))
                    n_mismatch += 1
                n_files += 1
        except Exception as e:
            failures.append((rel, f"ERROR: {e}"))
            n_mismatch += 1

    print()
    if failures:
        print(f"=== {n_mismatch} mismatched out of {n_files} compared ===")
        for rel, msg in failures[:50]:
            print(f"  [FAIL] {rel}: {msg}")
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")
        return 1
    else:
        print(f"=== ALL MATCH ({n_files} files) ===")
        return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
