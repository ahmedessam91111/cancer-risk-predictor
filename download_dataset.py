"""
Issue #13 -- obtain the exact training dataset, verified.

The dataset `cancer-risk-factors.csv` (2000 rows x 21 columns) is COMMITTED to
this repository, so a fresh `git clone` already ships it. This script is the
documented, runnable fallback for re-obtaining the *byte-identical* file from
the original public source, and for verifying whatever copy you have.

Upstream (confirmed byte-identical to the committed file, Issue #13):
    source : Tarek Masryo, "Cancer Risk Factors & Types (2,000 Rows)"
    primary: https://huggingface.co/datasets/tarekmasryo/cancer-risk-factors-data
             (file: data/cancer-risk-factors.csv)
    mirror : https://www.kaggle.com/datasets/tarekmasryo/cancer-risk-factors-dataset
    license: CC BY 4.0 (Attribution) -- free to share and adapt with attribution.

Usage:
    python download_dataset.py             # fetch if missing; verify the local file
    python download_dataset.py --force     # re-download and overwrite
    python download_dataset.py --verify-only   # check the local file without network

Exit codes:
    0  local file present and byte-identical to the recorded SHA-256
    1  dataset missing / hash mismatch / download failed
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = "cancer-risk-factors.csv"
DATASET_SHA256 = "01291f8babc1b1e5d8e8af5a4fcfd307b7ea5a2ed5255927c2d0ecc97ccb82a5"
DATASET_BYTES = 141851
DATASET_ROWS = 2000
DATASET_COLS = 21

UPSTREAM_URL = (
    "https://huggingface.co/datasets/tarekmasryo/cancer-risk-factors-data/"
    "resolve/main/data/cancer-risk-factors.csv"
)
# Kaggle mirror for humans (Kaggle requires login; the script downloads from HF):
MIRROR_URL = "https://www.kaggle.com/datasets/tarekmasryo/cancer-risk-factors-dataset"


def sha256_raw(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(local: Path) -> str:
    try:
        import pandas as pd
    except ImportError:
        return ("pandas not installed; shape not reported")
    df = pd.read_csv(local)
    return f"{df.shape[0]} rows x {df.shape[1]} columns"


def verify_local(local: Path) -> bool:
    if not local.exists():
        return False
    ok = local.stat().st_size == DATASET_BYTES and sha256_raw(local) == DATASET_SHA256
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {local.name}: {local.stat().st_size} bytes, "
          f"sha256 {sha256_raw(local)[:16]}...")
    if ok:
        print(f"       expected: {DATASET_BYTES} bytes, {DATASET_SHA256}")
        print(f"       shape   : {describe(local)}")
    return ok


def download(url: str, dest: Path) -> None:
    print(f"[..] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "cancer-risk-predictor/2 (reproducibility)"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"[..] downloaded {dest.stat().st_size} bytes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download and overwrite")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the local file only; no network access")
    args = ap.parse_args()

    local = BASE_DIR / DATASET_FILE

    if args.verify_only:
        if not local.exists():
            print(f"[FAIL] {DATASET_FILE} not found in {BASE_DIR}")
            return 1
        return 0 if verify_local(local) else 1

    if local.exists():
        if verify_local(local) and not args.force:
            print(f"[ok] {DATASET_FILE} already present and verified -- nothing to do "
                  f"(use --force to re-download, --verify-only to just re-check)")
            return 0
        if not args.force:
            print(f"[FAIL] {DATASET_FILE} exists but its hash does NOT match the "
                  f"recorded value (use --force to replace it)")
            return 1

    fd, tmp_name = tempfile.mkstemp(suffix=".csv", dir=str(BASE_DIR))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        download(UPSTREAM_URL, tmp)
        if not verify_local(tmp):
            print(f"[FAIL] downloaded file does not match recorded SHA-256 "
                  f"{DATASET_SHA256}; keeping your existing {DATASET_FILE} untouched")
            return 1
        tmp.replace(local)  # atomic on the same filesystem
    except Exception as exc:  # noqa: BLE001 -- surface any network/filesystem error
        print(f"[FAIL] {exc}")
        return 1
    finally:
        if tmp.exists():
            tmp.unlink()

    print(f"[ok] wrote {local.name} -- verified {DATASET_BYTES} bytes, "
          f"{DATASET_ROWS} rows x {DATASET_COLS} columns, sha256 {DATASET_SHA256}")
    print(f"[i]  mirror (login required): {MIRROR_URL}")
    print("[i]  license: CC BY 4.0 (Attribution) -- see docs/DATASET_PROVENANCE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())