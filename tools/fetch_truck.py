"""
Fetch the Tanks & Temples `truck` scene (3DGS's canonical COLMAP dataset) from
the HuggingFace mirror, with per-file size verification and resumable retries.

Usage:
    python fetch_truck.py --dest <dir> [--workers 6]

The file manifest (paths + exact byte sizes) is read from the mirror's dataset
tree API, so every downloaded file is verified, not just assumed.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "nvs-bench/tanksandtemples"
PREFIX = "truck"
API = f"https://hf-mirror.com/api/datasets/{REPO}/tree/main/{PREFIX}?recursive=true"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/{PREFIX}"

UA = {"User-Agent": "Mozilla/5.0 (dsh-3dgs-cpu-port)"}


def get_manifest():
    req = urllib.request.Request(API, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        items = json.load(r)
    out = []
    for it in items:
        if it.get("type") != "file":
            continue
        out.append((it["path"], int(it.get("size", 0))))
    return out


def fetch_one(rel_path, size, dest, retries=6):
    # rel_path looks like "truck/images/000001.jpg"
    rel = rel_path[len(PREFIX) + 1:]
    target = os.path.join(dest, rel.replace("/", os.sep))
    if os.path.exists(target) and (size == 0 or os.path.getsize(target) == size):
        return "skip", rel, size

    os.makedirs(os.path.dirname(target), exist_ok=True)
    url = f"{BASE}/{rel}"

    last_err = None
    for attempt in range(1, retries + 1):
        got = os.path.getsize(target) if os.path.exists(target) else 0
        headers = dict(UA)
        mode = "wb"
        if got and size and got < size:
            headers["Range"] = f"bytes={got}-"
            mode = "ab"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r, open(target, mode) as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            if size == 0 or os.path.getsize(target) == size:
                return "ok", rel, size
            last_err = f"size mismatch {os.path.getsize(target)} != {size}"
        except Exception as exc:                       # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2 * attempt, 8))

    return "fail", rel, last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print("reading manifest ...")
    manifest = get_manifest()
    total_bytes = sum(s for _, s in manifest)
    print(f"manifest: {len(manifest)} files, {total_bytes/1e6:.1f} MB")

    os.makedirs(args.dest, exist_ok=True)
    ok = skip = 0
    failures = []
    done_bytes = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, p, s, args.dest): (p, s) for p, s in manifest}
        for i, fut in enumerate(as_completed(futs), 1):
            status, rel, info = fut.result()
            if status == "ok":
                ok += 1
                done_bytes += info if isinstance(info, int) else 0
            elif status == "skip":
                skip += 1
                done_bytes += info if isinstance(info, int) else 0
            else:
                failures.append((rel, info))
            if i % 25 == 0 or i == len(manifest):
                el = time.time() - t0
                print(f"  {i}/{len(manifest)}  ok={ok} skipped={skip} failed={len(failures)}  "
                      f"{done_bytes/1e6:.1f}/{total_bytes/1e6:.1f} MB  {el:.0f}s")

    print(f"\nDONE ok={ok} skipped={skip} failed={len(failures)}")
    for rel, info in failures[:20]:
        print("  FAILED:", rel, info)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
