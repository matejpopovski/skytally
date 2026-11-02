#!/usr/bin/env python3
"""Fetch VisDrone2019-DET from the Ultralytics asset mirror and extract it under data/.

The archives are pinned by SHA-256. Only .jpg and .txt members in the expected folders
are extracted, so nothing else in an archive can reach the disk.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
ARCHIVES = {
    "VisDrone2019-DET-train.zip": (
        "86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84"
    ),
    "VisDrone2019-DET-val.zip": (
        "abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, expected: str, raw_dir: Path) -> Path:
    target = raw_dir / name
    if target.exists() and sha256(target) == expected:
        print(f"{name}: already present")
        return target
    print(f"{name}: downloading")
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(f"{BASE}/{name}") as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if (actual := sha256(partial)) != expected:
        partial.unlink()
        raise SystemExit(f"{name}: checksum mismatch, got {actual}")
    partial.replace(target)
    return target


def extract(archive: Path, out_dir: Path) -> None:
    split = archive.stem.removeprefix("VisDrone2019-DET-")
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            path = PurePosixPath(member.filename)
            allowed = {"images": ".jpg", "annotations": ".txt"}
            if len(path.parts) != 3 or allowed.get(path.parts[1]) != path.suffix:
                continue
            dest = out_dir / split / path.parts[1] / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    print(f"{split}: {count} files in {out_dir / split}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    raw_dir = args.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in ARCHIVES.items():
        archive = fetch(name, digest, raw_dir)
        extract(archive, args.data_dir / "visdrone")
        if not args.keep_archives:
            archive.unlink()


if __name__ == "__main__":
    main()
