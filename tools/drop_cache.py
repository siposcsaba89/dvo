"""Keeps the Linux page cache of a run's files small. Under WSL the VM's cache counts against Windows and is not
given back while the VM is busy: reading tens of GB of depth maps filled it to the VM limit and Windows ran out of
memory (2026-09-30, "low virtual memory: vmmemWSL 45 GB"). Without root, posix_fadvise(DONTNEED) drops the clean
cached pages of files we can open; done whenever the page cache exceeds --max_gb, until --pid exits.

  python tools/drop_cache.py RUN_DIR [--pid PID] [--max_gb 8]
"""

import argparse
import os
import time


def cached_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1]) / 2**20
    return 0.0


def drop(root):
    n = 0
    for folder, _, files in os.walk(root):
        for name in files:
            try:
                fd = os.open(os.path.join(folder, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    return n


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="folders whose files may be dropped from the cache")
    ap.add_argument("--pid", type=int, help="stop when this process exits")
    ap.add_argument("--max_gb", type=float, default=8.0)
    ap.add_argument("--every", type=float, default=20.0, help="s between checks")
    args = ap.parse_args()
    while args.pid is None or alive(args.pid):
        before = cached_gb()
        if before > args.max_gb:
            n = sum(drop(r) for r in args.roots)
            print(f"{time.strftime('%H:%M:%S')} cache {before:.1f} -> {cached_gb():.1f} GB ({n} files)", flush=True)
        if args.pid is None:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
