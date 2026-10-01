"""Trajectories (KITTI pose files) as one PLY for 3D viewers: per file a polyline (vertices + edges) in its own colour,
optionally with the body axes every --axes frames (x red, y green, z blue, 0.5 m).

  python tools/poses_to_ply.py OUT.ply poses.txt [diag/poses_loop.txt ...] [--axes 150]
"""

import argparse

import numpy as np

COLOURS = [(230, 60, 40), (40, 160, 230), (60, 200, 80), (230, 180, 40), (170, 80, 220)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("poses", nargs="+")
    ap.add_argument("--axes", type=int, default=0, help="draw the body axes every n frames (0 = off)")
    args = ap.parse_args()

    verts, cols, edges = [], [], []
    for k, f in enumerate(args.poses):
        T = np.loadtxt(f).reshape(-1, 3, 4)
        base = sum(len(v) for v in verts)
        verts.append(T[:, :, 3])
        cols.append(np.tile(COLOURS[k % len(COLOURS)], (len(T), 1)))
        edges.append(np.stack([np.arange(len(T) - 1), np.arange(1, len(T))], 1) + base)
        if args.axes > 0:
            for i in range(0, len(T), args.axes):
                o = T[i, :, 3]
                for a, c in zip(range(3), ((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                    n = sum(len(v) for v in verts)
                    verts.append(np.stack([o, o + 0.5 * T[i, :, a]]))
                    cols.append(np.array([c, c]))
                    edges.append(np.array([[n, n + 1]]))
        print(f"{f}: {len(T)} poses, colour {COLOURS[k % len(COLOURS)]}")
    v, c, e = np.concatenate(verts), np.concatenate(cols).astype(np.uint8), np.concatenate(edges).astype(np.int32)
    with open(args.out, "wb") as out:
        out.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n"
                   f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                   f"element edge {len(e)}\nproperty int vertex1\nproperty int vertex2\nend_header\n").encode())
        rec = np.zeros(len(v), [("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = v, c
        out.write(rec.tobytes())
        out.write(e.tobytes())
    print(f"wrote {args.out}: {len(v)} vertices, {len(e)} edges")


if __name__ == "__main__":
    main()
