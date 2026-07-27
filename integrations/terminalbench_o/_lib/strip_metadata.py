"""Strip identifying metadata from fixture files.

Used by setup_fixtures.sh scripts to remove EXIF / IPTC / XMP / ID3 / PDF
/Info dict / MP4 metadata-box tags that would otherwise let an agent
reverse-look up the source dataset by running `ffprobe` / `exiftool` on
fixtures.

Usage from a shell setup:
    python3 -m claw_bench._lib.strip_metadata path/to/file_or_dir [...]

Or programmatically:
    from claw_bench._lib.strip_metadata import strip_path
    strip_path("fixtures/")  # recurses

Supports:
    - JPEG / PNG / WEBP / TIFF      (Pillow re-save without EXIF)
    - PDF                            (pypdf metadata clear)
    - MP3 / FLAC / WAV / OGG         (ffmpeg -map_metadata -1)
    - MP4 / MOV / MKV / WEBM         (ffmpeg -map_metadata -1)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
PDF_EXT = {".pdf"}
AUDIO_EXT = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".opus"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def strip_image(path: Path) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        with Image.open(path) as im:
            data = list(im.getdata())
            mode = im.mode
            size = im.size
            fmt = im.format
            clean = Image.new(mode, size)
            clean.putdata(data)
            kw = {}
            if fmt in ("JPEG", "JPG"):
                kw = {"quality": 90, "optimize": True}
            tmp = path.with_suffix(path.suffix + ".tmp")
            clean.save(tmp, format=fmt, **kw)
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"  strip_image failed on {path}: {e}", file=sys.stderr)
        return False


def strip_pdf(path: Path) -> bool:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return False
    try:
        r = PdfReader(str(path))
        w = PdfWriter()
        for pg in r.pages:
            w.add_page(pg)
        w.add_metadata({"/Title": "", "/Author": "", "/Subject": "",
                        "/Keywords": "", "/Producer": "", "/Creator": ""})
        tmp = path.with_suffix(".tmp.pdf")
        with open(tmp, "wb") as fh:
            w.write(fh)
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"  strip_pdf failed on {path}: {e}", file=sys.stderr)
        return False


def strip_av(path: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    tmp = path.with_suffix(".tmp" + path.suffix)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-map_metadata", "-1",
        "-c", "copy", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp.replace(path)
        return True
    except subprocess.CalledProcessError as e:
        if tmp.exists():
            tmp.unlink()
        print(f"  strip_av failed on {path}: {e}", file=sys.stderr)
        return False


def strip_file(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in IMG_EXT:
        return strip_image(path)
    if ext in PDF_EXT:
        return strip_pdf(path)
    if ext in AUDIO_EXT or ext in VIDEO_EXT:
        return strip_av(path)
    return False


def strip_path(target: str | Path) -> int:
    p = Path(target)
    if p.is_file():
        return 1 if strip_file(p) else 0
    n = 0
    for child in p.rglob("*"):
        if child.is_file() and strip_file(child):
            n += 1
    return n


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: strip_metadata.py <path>...", file=sys.stderr)
        sys.exit(2)
    total = 0
    for arg in sys.argv[1:]:
        total += strip_path(arg)
    print(f"stripped metadata from {total} file(s)")
