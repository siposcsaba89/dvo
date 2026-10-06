"""Filter, thin and cut a large point cloud into viewable world-coordinate tiles.

Input is a binary little-endian PLY: cloud.ply (x, y, z, rgb) or points.ply (with source, camera, frame, distance,
observations, sigma, kept). Quality filters apply only where the attribute exists. Vertices used by edges
(the trajectory in cloud.ply) are dropped; open trajectory_loop.ply alongside instead. Coordinates are unchanged, so
any set of tiles opens together in CloudCompare / MeshLab and lines up.

  python tools/cloud_tiles.py R/diag/points.ply R/diag/tiles --kept --min_obs 8 --max_sigma 0.006 --voxel 0.05 --tile 100

Writes tile_<ix>_<iy>[_<iz>].ply (x = ix*tile .. (ix+1)*tile, same for y, z with --tile_z), overview.ply (whole area at --overview_voxel)
and tiles.csv (bounds and point count per tile).
"""

import argparse
import os

import numpy as np

PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
             "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
             "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}
PLY_NAMES = {"<f4": "float", "<f8": "double", "u1": "uchar", "i1": "char", "<u2": "ushort", "<i2": "short",
             "<u4": "uint", "<i4": "int"}


def read_ply(path):
    """Vertex array (memory-mapped) and the edge-referenced vertex indices of a binary little-endian PLY."""
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            head += line
    lines = head.decode().splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"{path}: only binary little-endian PLY")
    elements = []
    for line in lines:
        e = line.split()
        if e[0] == "element":
            elements.append([e[1], int(e[2]), []])
        elif e[0] == "property":
            if e[1] == "list":
                raise ValueError(f"{path}: list properties are not supported")
            elements[-1][2].append((e[2], PLY_TYPES[e[1]]))
    offset = len(head)
    vertices, used = None, np.zeros(0, np.int64)
    for name, count, props in elements:
        dtype = np.dtype(props)
        data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
        if name == "vertex":
            vertices = data
        elif name == "edge":
            used = np.unique(np.concatenate([np.asarray(data[p[0]], np.int64) for p in props[:2]]))
        offset += count * dtype.itemsize
    return vertices, used


def write_ply(path, data):
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    header += [f"property {PLY_NAMES[data.dtype[n].str.replace('|', '')]} {n}" for n in data.dtype.names]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        data.tofile(f)


def voxel_thin(xyz, voxel, priority):
    """Indices of one point per voxel, the one with the highest priority (first one without)."""
    q = np.floor(xyz / voxel).astype(np.int64)
    q -= q.min(axis=0)
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    order = np.lexsort((-priority, key)) if priority is not None else np.argsort(key, kind="stable")
    k = key[order]
    first = np.ones(len(k), bool)
    first[1:] = k[1:] != k[:-1]
    return np.sort(order[first])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply")
    ap.add_argument("out_dir")
    ap.add_argument("--kept", action="store_true", help="only points that passed the neighbour filter (kept == 1)")
    ap.add_argument("--min_obs", type=int, default=0, help="minimum observations")
    ap.add_argument("--max_sigma", type=float, default=0, help="maximum relative depth sigma (0 = off)")
    ap.add_argument("--max_distance", type=float, default=0, help="maximum distance from the host camera, m (0 = off)")
    ap.add_argument("--sources", type=int, nargs="+", help="point sources to keep (0 active, 1 candidate, 2 semi-dense)")
    ap.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"), help="host frame range")
    ap.add_argument("--bbox", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"), help="world box")
    ap.add_argument("--voxel", type=float, default=0, help="keep one point per voxel of this size, m (0 = off)")
    ap.add_argument("--tile", type=float, default=100, help="tile edge in x and y, m (0 = no tiles)")
    ap.add_argument("--tile_z", type=float, default=0, help="also cut in z with this tile height, m (garage levels)")
    ap.add_argument("--overview_voxel", type=float, default=0.5, help="voxel of overview.ply, m (0 = no overview)")
    ap.add_argument("--attributes", action="store_true",
                    help="keep the scalar attributes in the output (colour by them in CloudCompare); default xyz + rgb")
    args = ap.parse_args()

    v, used = read_ply(args.ply)
    names = v.dtype.names
    keep = np.ones(len(v), bool)
    keep[used] = False
    print(f"{args.ply}: {len(v)} vertices, {len(used)} used by edges")

    def require(name):
        if name not in names:
            raise SystemExit(f"{args.ply} has no '{name}' property (use points.ply for quality filters)")
        return np.asarray(v[name])

    def apply(label, mask):
        nonlocal keep
        before = int(keep.sum())
        keep &= mask
        print(f"  {label}: {before} -> {int(keep.sum())}")

    if args.kept:
        apply("kept", require("kept") != 0)
    if args.min_obs:
        apply(f"observations >= {args.min_obs}", require("observations") >= args.min_obs)
    if args.max_sigma:
        apply(f"sigma <= {args.max_sigma}", require("sigma") <= args.max_sigma)
    if args.max_distance:
        apply(f"distance <= {args.max_distance}", require("distance") <= args.max_distance)
    if args.sources:
        apply(f"source in {args.sources}", np.isin(require("source"), args.sources))
    if args.frames:
        f = require("frame")
        apply(f"frame {args.frames[0]}..{args.frames[1]}", (f >= args.frames[0]) & (f <= args.frames[1]))
    xyz = np.stack([np.asarray(v[c]) for c in "xyz"], axis=1)
    if args.bbox:
        lo, hi = np.array(args.bbox[:3]), np.array(args.bbox[3:])
        apply("bbox", np.all((xyz >= lo) & (xyz <= hi), axis=1))

    idx = np.flatnonzero(keep)
    xyz = xyz[idx]
    priority = np.asarray(v["observations"])[idx] if "observations" in names else None
    if args.voxel:
        sel = voxel_thin(xyz, args.voxel, priority)
        print(f"  voxel {args.voxel} m: {len(idx)} -> {len(sel)}")
        idx, xyz = idx[sel], xyz[sel]
        if priority is not None:
            priority = priority[sel]

    fields = list(names) if args.attributes else [n for n in ("x", "y", "z", "red", "green", "blue") if n in names]
    out = np.empty(len(idx), np.dtype([(n, v.dtype[n]) for n in fields]))
    for n in fields:
        out[n] = v[n][idx]
    os.makedirs(args.out_dir, exist_ok=True)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    print(f"{len(out)} points, extent x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f}, z {lo[2]:.1f}..{hi[2]:.1f}")

    if args.overview_voxel:
        sel = voxel_thin(xyz, args.overview_voxel, priority)
        write_ply(os.path.join(args.out_dir, "overview.ply"), out[sel])
        print(f"overview.ply: {len(sel)} points ({args.overview_voxel} m voxel)")

    if not args.tile:
        write_ply(os.path.join(args.out_dir, "cloud.ply"), out)
        print(f"cloud.ply: {len(out)} points")
        return
    t = np.floor(xyz / np.array([args.tile, args.tile, args.tile_z or 1.0])).astype(np.int64)
    if not args.tile_z:
        t[:, 2] = 0
    span = t.max(axis=0) - t.min(axis=0) + 1
    tt = t - t.min(axis=0)
    key = (tt[:, 0] * span[1] + tt[:, 1]) * span[2] + tt[:, 2]
    order = np.argsort(key, kind="stable")
    starts = np.flatnonzero(np.r_[True, key[order][1:] != key[order][:-1]])
    ends = np.r_[starts[1:], len(order)]
    with open(os.path.join(args.out_dir, "tiles.csv"), "w") as csv:
        csv.write("file,x0,y0,z0,x1,y1,z1,points\n")
        for s, e in zip(starts, ends):
            ix, iy, iz = t[order[s]]
            name = f"tile_{ix}_{iy}_{iz}.ply" if args.tile_z else f"tile_{ix}_{iy}.ply"
            write_ply(os.path.join(args.out_dir, name), out[np.sort(order[s:e])])
            x0, y0 = ix * args.tile, iy * args.tile
            z0, z1 = (iz * args.tile_z, (iz + 1) * args.tile_z) if args.tile_z else (-np.inf, np.inf)
            csv.write(f"{name},{x0:g},{y0:g},{z0:g},{x0 + args.tile:g},{y0 + args.tile:g},{z1:g},{e - s}\n")
    sizes = ends - starts
    print(f"{len(starts)} tiles of {args.tile:g} m, {sizes.min()}..{sizes.max()} points (median {int(np.median(sizes))})")


if __name__ == "__main__":
    main()
