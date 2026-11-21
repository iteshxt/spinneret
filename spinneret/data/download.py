"""Fetch and unpack Spider 1.0 (CC BY-SA 4.0) into ``data/raw/spider``."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from spinneret.data.spider import missing_files

# Official Google Drive file linked from https://yale-lily.github.io/spider
GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
OFFICIAL_URL = (
    f"https://drive.usercontent.google.com/download?id={GDRIVE_ID}&export=download&confirm=t"
)
# SHA-256 of spider_data.zip (205,800,266 bytes); matches the HAL-9001/spider-databases mirror.
EXPECTED_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
ZIP_PREFIX = "spider_data/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
        total = int(resp.headers.get("Content-Length") or 0)
        with tmp.open("wb") as out, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def extract(zip_path: Path, spider_dir: Path) -> int:
    """Unpack ``spider_data/*`` into ``spider_dir``, skipping macOS junk and unsafe paths."""
    spider_dir.mkdir(parents=True, exist_ok=True)
    root = spider_dir.resolve()
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name.startswith(ZIP_PREFIX) or name.endswith("/"):
                continue
            rel = name[len(ZIP_PREFIX) :]
            if not rel or "__MACOSX" in rel or rel.endswith(".DS_Store"):
                continue
            target = (spider_dir / rel).resolve()
            if root not in target.parents:  # zip-slip guard
                raise ValueError(f"unsafe path in archive: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def download_spider(raw_dir: Path, *, zip_path: Path | None = None, force: bool = False) -> Path:
    """Download (unless ``zip_path`` is given), verify and extract Spider.

    Returns the extracted directory (``raw_dir / "spider"``).
    """
    spider_dir = raw_dir / "spider"
    if not force and not missing_files(spider_dir):
        return spider_dir

    if zip_path is None:
        zip_path = raw_dir / "download" / "spider_data.zip"
        if force or not zip_path.is_file() or sha256(zip_path) != EXPECTED_SHA256:
            fetch(OFFICIAL_URL, zip_path)

    actual = sha256(zip_path)
    if actual != EXPECTED_SHA256:
        raise ValueError(
            f"checksum mismatch for {zip_path}: expected {EXPECTED_SHA256}, got {actual}"
        )
    extract(zip_path, spider_dir)
    missing = missing_files(spider_dir)
    if missing:
        raise FileNotFoundError(f"Spider extraction incomplete, missing: {', '.join(missing)}")
    return spider_dir
