"""Quick look at one or more trajectories (KITTI pose files, one pose per input frame): top view coloured by height
and height over time, e.g. odometry vs loop-closed poses. Levels of a multi-storey garage show as height plateaus.

  python tools/plot_trajectory.py OUT.png poses.txt [diag/poses_loop.txt ...] [--fps 15]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("poses", nargs="+")
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()

    tracks = [(f, np.loadtxt(f).reshape(-1, 3, 4)[:, :, 3]) for f in args.poses]
    n = len(tracks)
    fig, axes = plt.subplots(1, n + 1, figsize=(6 * n + 7, 6))
    lo = min(p[:, 2].min() for _, p in tracks)
    hi = max(p[:, 2].max() for _, p in tracks)
    for ax, (name, p) in zip(axes, tracks):
        sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], s=1, cmap="viridis", vmin=lo, vmax=hi)
        ax.plot(p[0, 0], p[0, 1], "r^", ms=8)
        ax.set_aspect("equal")
        length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        ax.set_title(f"{name.split('/')[-1]}: {length:.0f} m, colour = height", fontsize=9)
        ax.set_xlabel("x m")
        ax.set_ylabel("y m")
    fig.colorbar(sc, ax=axes[:n], shrink=0.8, label="z m")
    for name, p in tracks:
        axes[n].plot(np.arange(len(p)) / args.fps, p[:, 2], lw=1, label=name.split("/")[-1])
    axes[n].set_xlabel("time s")
    axes[n].set_ylabel("z m")
    axes[n].set_title("height over time", fontsize=9)
    axes[n].grid(alpha=0.3)
    axes[n].legend(fontsize=8)
    fig.savefig(args.out, dpi=80, bbox_inches="tight")
    for name, p in tracks:
        print(f"{name}: {len(p)} frames, {np.linalg.norm(np.diff(p, axis=0), axis=1).sum():.0f} m, "
              f"z {p[:, 2].min():.2f} .. {p[:, 2].max():.2f} m, end at {np.round(p[-1], 2)}")


if __name__ == "__main__":
    main()
