"""Level a run: without IMU the world "up" is the body up of the first frame, so a start on a slope tilts the whole
map (Chili 2026-09-22: 2.3 deg, level floors rising 0.7 m over 40 m). Up is estimated as the robust mean body up
axis over the drive (frames more than --max_dev deg off it, e.g. ramps, are left out, iteratively), and the poses and
point clouds are rotated about the origin by the smallest rotation that makes it vertical. Files are rewritten in
place; the original poses are kept as <poses>.unlevelled.

  python tools/level_run.py POSES.txt [CLOUD.ply ...]
"""

import argparse
import os
import shutil

import numpy as np

PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
             "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
             "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}


def estimate_up(R_wb, max_dev):
    up = np.median(R_wb[:, :, 2], axis=0)
    up /= np.linalg.norm(up)
    for _ in range(5):
        keep = R_wb[:, :, 2] @ up > np.cos(np.radians(max_dev))
        up = R_wb[keep, :, 2].mean(axis=0)
        up /= np.linalg.norm(up)
    return up, keep


def rotation_to_z(u):
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(u, z), float(u @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V / (1 + c)


def rotate_ply(path, R):
    """Rotates x, y, z of the vertex element of a binary little-endian PLY in place (other elements untouched)."""
    with open(path, "r+b") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        lines = head.decode().splitlines()
        if "format binary_little_endian 1.0" not in lines:
            raise ValueError(f"{path}: only binary little-endian PLY")
        elements, current = [], None
        for line in lines:
            e = line.split()
            if e[0] == "element":
                current = [e[1], int(e[2]), []]
                elements.append(current)
            elif e[0] == "property":
                if e[1] == "list":
                    raise ValueError(f"{path}: list properties before the vertices are not supported")
                current[2].append((e[2], PLY_TYPES[e[1]]))
        name, count, props = elements[0]
        if name != "vertex" or [p[0] for p in props[:3]] != ["x", "y", "z"]:
            raise ValueError(f"{path}: the first element must be vertex with x, y, z first")
        dtype = np.dtype(props)
        data = np.fromfile(f, dtype=dtype, count=count)
        xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64) @ R.T
        for k, a in enumerate("xyz"):
            data[a] = xyz[:, k]
        f.seek(len(head))
        data.tofile(f)
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poses", help="KITTI body poses (T_w_b), rewritten in place")
    ap.add_argument("clouds", nargs="*", help="binary PLYs in the same world frame, rotated in place")
    ap.add_argument("--max_dev", type=float, default=3.0, help="deg; frames tilted more than this are left out")
    args = ap.parse_args()

    backup = args.poses + ".unlevelled"
    if os.path.exists(backup):
        raise SystemExit(f"{backup} exists: this run is levelled already")
    T = np.loadtxt(args.poses).reshape(-1, 3, 4)
    up, keep = estimate_up(T[:, :, :3], args.max_dev)
    R = rotation_to_z(up)
    tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
    print(f"up from {keep.sum()} of {len(T)} frames: {np.round(up, 5)}, map tilted {tilt:.2f} deg")
    shutil.copy(args.poses, backup)
    L = np.concatenate([R, np.zeros((3, 1))], 1)
    out = np.einsum("ij,njk->nik", R, T)  # rotation and translation: T' = [R 0] [R_wb t_wb] = [R R_wb, R t_wb]
    np.savetxt(args.poses, out.reshape(len(T), 12), fmt="%.9e")
    body_up = out[keep, :, 2]
    print(f"after: body up vs z on those frames median {np.degrees(np.median(np.arccos(np.clip(body_up[:, 2], -1, 1)))):.2f} deg")
    for c in args.clouds:
        print(f"{c}: {rotate_ply(c, R)} vertices rotated")


if __name__ == "__main__":
    main()
