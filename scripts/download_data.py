"""
Download the Synchronised (Uncategorised) IO-VNBD V-Dataset and S-Dataset
from GitHub Git-LFS storage into data/raw/.

The repo stores CSVs via Git LFS, so the normal raw URL only returns a ~130 byte
pointer. Real bytes are served from media.githubusercontent.com/media/... .
We validate every download to make sure we got real CSV content, not a pointer.
"""
import os
import sys
import time
import json
import urllib.parse
import urllib.request

REPO = "onyekpeu/IO-VNBD"
BRANCH = "master"
BASE_DIR = "Synchronised V abd S datasets/Uncategorised IOVNB Dataset"
SUBSETS = ["V-Dataset", "S-Dataset"]

API = "https://api.github.com/repos/{repo}/contents/{path}"
MEDIA = "https://media.githubusercontent.com/media/{repo}/{branch}/{path}"

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def http_get(url, binary=False, retries=4):
    hdr = {"User-Agent": "sih-idr-downloader"}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return data if binary else data.decode("utf-8")
        except Exception as e:  # noqa
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def list_files(subset):
    path = f"{BASE_DIR}/{subset}"
    url = API.format(repo=REPO, path=urllib.parse.quote(path))
    items = json.loads(http_get(url))
    return sorted(x["name"] for x in items if x["type"] == "file" and x["name"].endswith(".csv"))


def looks_like_pointer(b):
    head = b[:200].lstrip()
    return head.startswith(b"version https://git-lfs")


def main():
    total = 0
    for subset in SUBSETS:
        out_dir = os.path.join(OUT_ROOT, subset)
        os.makedirs(out_dir, exist_ok=True)
        names = list_files(subset)
        print(f"[{subset}] {len(names)} files")
        for i, name in enumerate(names, 1):
            dest = os.path.join(out_dir, name)
            if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                continue
            path = f"{BASE_DIR}/{subset}/{name}"
            url = MEDIA.format(repo=REPO, branch=BRANCH,
                               path=urllib.parse.quote(path))
            data = http_get(url, binary=True)
            if looks_like_pointer(data) or len(data) < 500:
                print(f"  !! POINTER/EMPTY for {name} ({len(data)} bytes)")
                continue
            with open(dest, "wb") as f:
                f.write(data)
            total += 1
            if i % 10 == 0 or i == len(names):
                print(f"  {subset}: {i}/{len(names)} done")
    print(f"Downloaded {total} new files into {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()
