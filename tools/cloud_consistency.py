"""Free-space consistency of a close_loops point cloud (--points-out), without ground truth.

Every densify host image (keyframe, camera) has its own semi-dense depth: the distances of the points it hosts. A
point P is tested in every other host image v that sees it (within --radius): v's depth at P's pixel (the nearest of
v's own points within a few pixels) is compared with P's distance from v.
  support:  v measured the same distance (within --tol)
  through:  v measured a surface clearly behind P, i.e. v saw through P: P lies in free space (a floater)
  behind:   v measured something clearly in front of P (occlusion, or P is a mirror point under a glossy floor)
A point is a floater when through >= 2 and through > support.

  python tools/cloud_consistency.py --rig RIG.yaml --cameras CAM... --poses poses_loop.txt --points points.ply
"""

import argparse
import os

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree

PLY_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "char": "i1", "int": "<i4", "uint": "<u4",
             "short": "<i2", "ushort": "<u2"}
SOURCES = ["active", "candidate", "semidense"]


def read_ply(path):
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[2])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, PLY_TYPES[t]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=n)


def write_ply(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                 % len(xyz)).encode())
        a = np.empty(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        a["p"], a["c"] = xyz, rgb
        a.tofile(f)


def load_rig(path, names, scale):
    rig = yaml.safe_load(open(path))
    base = os.path.dirname(path)
    cams = []
    for c in rig["cameras"]:
        if c["name"] not in names:
            continue
        k = yaml.safe_load(open(os.path.join(base, c["camera"])))
        s = scale * (c.get("image_width", k["width"]) / k["width"])
        w, h = round(round(k["width"] * c.get("image_width", k["width"]) / k["width"]) * scale), 0
        h = round(round(k["height"] * c.get("image_width", k["width"]) / k["width"]) * scale)
        T = np.eye(4)
        T[:3, :3] = np.array(c["T_body_camera"]["rotation_matrix"]).reshape(3, 3)
        T[:3, 3] = c["T_body_camera"]["translation"]
        cams.append(dict(name=c["name"], fx=k["fx"] * s, fy=k["fy"] * s, cx=(k["cx"] + 0.5) * s - 0.5,
                         cy=(k["cy"] + 0.5) * s - 0.5, alpha=k.get("alpha", 0.0), beta=k.get("beta", 1.0),
                         w=w - w % 16, h=h - h % 16, T_b_c=T))
    if len(cams) != len(names):
        raise SystemExit("unknown camera in --cameras")
    return cams


def project(cam, X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    d = np.sqrt(cam["beta"] * (x * x + y * y) + z * z)
    den = cam["alpha"] * d + (1 - cam["alpha"]) * z
    a = cam["alpha"]
    w = a / (1 - a) if a <= 0.5 else (1 - a) / a
    ok = (z > -w * d) & (den > 1e-9) & (z > 0.05)
    den = np.where(ok, den, 1.0)
    return cam["fx"] * x / den + cam["cx"], cam["fy"] * y / den + cam["cy"], ok


def load_poses(path):
    m = np.loadtxt(path).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(m), 1, 1))
    T[:, :3, :] = m
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", required=True)
    ap.add_argument("--cameras", nargs="+", required=True, help="densify cameras (close_loops --densify-cameras)")
    ap.add_argument("--poses", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--reference", help="points.ply whose host images give the depth maps (default: --points), so "
                    "that clouds of the same run are tested against the same views")
    ap.add_argument("--anchor_poses", help="poses of the run that made --points (e.g. odometry before loop closure): "
                    "each point is moved with its host frame into --poses, so that drift does not count")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=15.0, help="views test points within this distance, m")
    ap.add_argument("--tol", type=float, default=0.05, help="relative distance agreement")
    ap.add_argument("--margin", type=float, default=0.10, help="relative distance for through / behind")
    ap.add_argument("--window", type=int, default=2, help="pixel radius of a view's depth lookup")
    ap.add_argument("--sample", type=int, default=400000, help="tested points (0 = all)")
    ap.add_argument("--views", choices=["all", "even", "odd"], default="all",
                    help="use only every other host frame as views (hold-out tests of a free-space filter)")
    ap.add_argument("--exclude_npz", help="leave out the points flagged as floaters in this --out_npz (filter test)")
    ap.add_argument("--out_ply", help="tested points coloured: green support, red floater, blue behind, grey unknown")
    ap.add_argument("--out_npz", help="per tested point: index, support, through, behind")
    args = ap.parse_args()

    cams = load_rig(args.rig, args.cameras, args.scale)
    T_w_b = load_poses(args.poses)
    P = read_ply(args.points)
    xyz = np.stack([P["x"], P["y"], P["z"]], 1).astype(np.float64)
    if args.anchor_poses:
        T_a = load_poses(args.anchor_poses)
        M = T_w_b[P["frame"]] @ np.linalg.inv(T_a[P["frame"]])
        xyz = np.einsum("nij,nj->ni", M[:, :3, :3], xyz) + M[:, :3, 3]
    semi = P["source"] == 2
    print(f"{len(P)} points ({semi.sum()} semi-dense, {(P['source'] == 0).sum()} active)")

    # Host images: semi-dense points carry the densify camera index.
    host_key = P["frame"].astype(np.int64) * 64 + P["camera"]
    Q = read_ply(args.reference) if args.reference else P
    ref_key = Q["frame"].astype(np.int64) * 64 + Q["camera"]
    ref_semi = Q["source"] == 2
    views = np.unique(ref_key[ref_semi])
    if args.views != "all":
        frames = np.unique(views // 64)
        keep_frames = frames[(np.arange(len(frames)) % 2) == (0 if args.views == "even" else 1)]
        views_sel = views[np.isin(views // 64, keep_frames)]
    else:
        views_sel = views
    order = np.argsort(ref_key[ref_semi], kind="stable")
    semi_idx = np.nonzero(ref_semi)[0][order]
    starts = np.searchsorted(ref_key[semi_idx], views)
    ends = np.searchsorted(ref_key[semi_idx], views, side="right")

    rng = np.random.default_rng(0)
    test = np.arange(len(P)) if args.sample <= 0 or args.sample >= len(P) else np.sort(
        rng.choice(len(P), args.sample, replace=False))
    if args.exclude_npz:
        e = np.load(args.exclude_npz)
        flagged = e["index"][(e["through"] >= 2) & (e["through"] > e["support"])]
        before = len(test)
        test = test[~np.isin(test, flagged)]
        print(f"excluded {before - len(test)} of {before} tested points flagged in {args.exclude_npz}")
    tree = cKDTree(xyz[test])
    sup = np.zeros(len(test), np.int32)
    thr = np.zeros(len(test), np.int32)
    beh = np.zeros(len(test), np.int32)
    # Histogram of log(view depth / point depth) over all tests: the peak width is the depth precision (two
    # independent estimates), the tails are wrong points and occlusions.
    edges = np.linspace(-0.3, 0.3, 241)
    hist = np.zeros(len(edges) - 1)
    # Split by frame gap between the point's host and the view: same frame (other camera: calibration), within the
    # tracing window, the same pass, revisits (loop alignment).
    gaps = [(0, 1, "same frame"), (1, 31, "gap 1-30"), (31, 300, "gap 31-299"), (300, 10 ** 9, "gap 300+")]
    dists = [(0, 3), (3, 6), (6, 15)]
    ghist = np.zeros((len(gaps), len(dists), len(edges) - 1))
    frame_of = P["frame"].astype(np.int64)
    pair_log = []  # (host key, view key, log ratio) of the smooth precision tests
    views_sel_set = set(views_sel.tolist())
    k = 2 * args.window + 1
    kernel = np.ones((k, k), np.uint8)
    for vi, key in enumerate(views):
        if args.views != "all" and key not in views_sel_set:
            continue
        frame, c = int(key // 64), int(key % 64)
        cam = cams[c]
        T_w_c = T_w_b[frame] @ cam["T_b_c"]
        T_c_w = np.linalg.inv(T_w_c)
        own = semi_idx[starts[vi]:ends[vi]]
        u0 = np.rint(Q["u"][own]).astype(int)
        v0 = np.rint(Q["v"][own]).astype(int)
        inside = (u0 >= 0) & (v0 >= 0) & (u0 < cam["w"]) & (v0 < cam["h"])
        depth = np.full((cam["h"], cam["w"]), np.inf, np.float32)
        np.minimum.at(depth, (v0[inside], u0[inside]), Q["distance"][own][inside].astype(np.float32))
        depth = cv2.erode(depth, kernel)  # nearest of the view's own points within the window

        cand = np.array(tree.query_ball_point(T_w_c[:3, 3], args.radius), dtype=np.int64)
        if len(cand) == 0:
            continue
        if not args.anchor_poses:
            cand = cand[host_key[test[cand]] != key]
        Xc = xyz[test[cand]] @ T_c_w[:3, :3].T + T_c_w[:3, 3]
        u, v, ok = project(cam, Xc)
        ui, vi_ = np.rint(u).astype(int), np.rint(v).astype(int)
        ok &= (ui >= 0) & (vi_ >= 0) & (ui < cam["w"]) & (vi_ < cam["h"])
        cand, Xc, ui, vi_ = cand[ok], Xc[ok], ui[ok], vi_[ok]
        dv = depth[vi_, ui]
        known = np.isfinite(dv)
        cand, dp, dv = cand[known], np.linalg.norm(Xc[known], axis=1), dv[known]
        r = dv / dp
        hist += np.histogram(np.log(r), edges)[0]

        # Precision: the view's depth at P's exact sub-pixel position from a plane fitted to its own points within
        # 3 px (smooth surfaces only), so that the pixel sampling of slanted surfaces does not count as error.
        uvo = np.stack([Q["u"][own][inside], Q["v"][own][inside]], 1).astype(np.float64)
        do = Q["distance"][own][inside].astype(np.float64)
        if len(uvo) >= 8:
            pq = np.stack([u[ok][known], v[ok][known]], 1)
            dist_nb, nb = cKDTree(uvo).query(pq, k=8, distance_upper_bound=3.0)
            valid = np.isfinite(dist_nb)
            enough = valid.sum(1) >= 5
            nb, valid, pq_e = np.where(valid, nb, 0)[enough], valid[enough], pq[enough]
            A = np.concatenate([np.ones(nb.shape + (1,)), uvo[nb] - pq_e[:, None, :]], 2) * valid[..., None]
            y = do[nb] * valid
            AtA = np.einsum("nki,nkj->nij", A, A) + 1e-9 * np.eye(3)
            coef = np.linalg.solve(AtA, np.einsum("nki,nk->ni", A, y)[..., None])[..., 0]
            fit = np.einsum("nki,ni->nk", A, coef)
            rms = np.sqrt(((fit - y) ** 2 * valid).sum(1) / valid.sum(1)) / np.maximum(coef[:, 0], 1e-6)
            smooth = (rms < 0.005) & (coef[:, 0] > 0)
            rp = coef[smooth, 0] / dp[enough][smooth]
            gap = np.abs(frame_of[test[cand[enough][smooth]]] - frame)
            dps = dp[enough][smooth]
            sel = test[cand[enough][smooth]]
            pair_log.append(np.stack([host_key[sel].astype(np.float64), np.full(len(rp), float(key)), np.log(rp), dps,
                                      P["v"][sel].astype(np.float64), v[ok][known][enough][smooth]], 1))
            for gi, (lo, hi, _) in enumerate(gaps):
                for di, (dlo, dhi) in enumerate(dists):
                    m = (gap >= lo) & (gap < hi) & (dps >= dlo) & (dps < dhi)
                    ghist[gi, di] += np.histogram(np.log(rp[m]), edges)[0]
        sup[cand] += np.abs(r - 1) < args.tol
        thr[cand] += (r > 1 + args.margin) & (dv - dp > 0.1)
        beh[cand] += (r < 1 - args.margin) & (dp - dv > 0.1)
        if vi % 500 == 0:
            print(f"view {vi} / {len(views)}", flush=True)

    floater = (thr >= 2) & (thr > sup)
    behind = (beh >= 2) & (beh > sup) & ~floater
    seen = sup + thr + beh > 0
    confirmed = (sup >= 2) & ~floater

    def report(label, m):
        n = max(int(m.sum()), 1)
        s = max(int((m & seen).sum()), 1)
        print(f"  {label:28s} {int(m.sum()):9d}  tested {100 * (m & seen).sum() / n:5.1f} %  confirmed "
              f"{100 * (confirmed & m).sum() / s:5.1f} %  floater {100 * (floater & m).sum() / s:5.2f} %  "
              f"behind {100 * (behind & m).sum() / s:5.2f} %")

    src = P["source"][test]
    kept = P["kept"][test] > 0
    obs = P["observations"][test]
    dist = P["distance"][test]
    print(f"tested {len(test)} points against {len(views)} host images (radius {args.radius} m)")
    report("all", np.ones(len(test), bool))
    report("kept (neighbour filter)", kept)
    for s in np.unique(src):
        report(f"source {SOURCES[s]}", src == s)
    for c, cam in enumerate(cams):
        report(f"semidense host {cam['name']}", (src == 2) & (P["camera"][test] == c))
    for lo, hi in [(0, 2), (2, 4), (4, 8), (8, 15), (15, 100)]:
        report(f"distance {lo}-{hi} m", (dist >= lo) & (dist < hi))
    for lo, hi in [(0, 4), (4, 8), (8, 16), (16, 32), (32, 10 ** 6)]:
        report(f"observations {lo}-{hi - 1}", (obs >= lo) & (obs < hi))

    lr = 0.5 * (edges[1:] + edges[:-1])
    total = hist.sum()
    within = lambda x: hist[np.abs(lr) < x].sum() / total * 100
    print(f"view tests {int(total)}: |log ratio| < 1 % {within(0.01):.1f} %, < 2 % {within(0.02):.1f} %, "
          f"< 5 % {within(0.05):.1f} %, < 10 % {within(0.10):.1f} %")
    core = np.abs(lr) < 0.05
    for gi, (_, _, label) in enumerate(gaps):
        for di, (dlo, dhi) in enumerate([(0, 1000)] + dists):
            h = ghist[gi].sum(0) if di == 0 else ghist[gi, di - 1]
            if h.sum() == 0:
                continue
            rms = np.sqrt((h[core] * lr[core] ** 2).sum() / max(h[core].sum(), 1))
            bias = (h[core] * lr[core]).sum() / max(h[core].sum(), 1)
            dl = "all" if di == 0 else f"{dlo}-{dhi} m"
            print(f"  {label:12s} {dl:7s} {int(h.sum()):9d} tests: < 1 % {h[np.abs(lr) < 0.01].sum() / h.sum() * 100:5.1f} %, "
                  f"< 2 % {h[np.abs(lr) < 0.02].sum() / h.sum() * 100:5.1f} %, core rms {rms * 100:.2f} %, "
                  f"bias {bias * 100:+.2f} %")
    if pair_log:
        # Error structure: a pose or timing error shifts all points of a (host, view) pair alike; depth noise does not.
        a = np.concatenate(pair_log)
        a = a[np.abs(a[:, 2]) < 0.05]
        hf, vf = a[:, 0] // 64, a[:, 1] // 64
        a = a[np.abs(hf - vf) >= 31]
        keys = a[:, 0] * 1e6 + a[:, 1]
        u, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        med = np.array([np.median(a[inv == i, 2]) if c >= 20 else np.nan for i, c in enumerate(cnt)])
        good = np.isfinite(med)
        within = a[:, 2] - med[inv]
        m = good[inv]
        print(f"pairs gap >= 31 with 20+ tests: {good.sum()}, their median log ratio: spread (rms) "
              f"{np.sqrt(np.nanmean(med[good] ** 2)) * 100:.2f} %; scatter within pairs (rms) "
              f"{np.sqrt(np.mean(within[m] ** 2)) * 100:.2f} %; all tests rms {np.sqrt(np.mean(a[m, 2] ** 2)) * 100:.2f} %")
    if args.out_npz:
        np.savez_compressed(args.out_npz, pairs=np.concatenate(pair_log) if pair_log else np.zeros((0, 6)), index=test, support=sup, through=thr, behind=beh, hist=hist,
                            edges=edges, ghist=ghist)
    if args.out_ply:
        rgb = np.full((len(test), 3), 128, np.uint8)
        rgb[confirmed] = (40, 200, 40)
        rgb[behind] = (40, 80, 255)
        rgb[floater] = (255, 30, 30)
        write_ply(args.out_ply, xyz[test], rgb)


if __name__ == "__main__":
    main()
