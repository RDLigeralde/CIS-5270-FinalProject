"""
Fetch dataset files into data_files/.

Sources:
  (default)      Google Drive folder shared with the project
  --local PATH   Copy from a local directory

File names on Drive match local names exactly.
"""

import os
import shutil
import argparse
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1KClYlFqW6cqP0cSsjBuBWSlOw2j08zPB"


def fetch_from_drive() -> None:
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown not installed. Run: pip install gdown")

    os.makedirs(DATA_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        gdown.download_folder(GDRIVE_FOLDER_URL, output=tmp, quiet=False)
        # gdown places files in tmp/<gdrive-folder-name>/
        subdirs = [e for e in os.scandir(tmp) if e.is_dir()]
        src = subdirs[0].path if subdirs else tmp
        for entry in sorted(os.scandir(src), key=lambda e: e.name):
            if entry.is_file():
                dest = os.path.join(DATA_DIR, entry.name)
                shutil.copy2(entry.path, dest)
                print(f"  {entry.name} -> {dest}")


def fetch_from_local(src_dir: str) -> None:
    if not os.path.isdir(src_dir):
        raise SystemExit(f"Not a directory: {src_dir}")
    os.makedirs(DATA_DIR, exist_ok=True)
    copied = 0
    for entry in sorted(os.scandir(src_dir), key=lambda e: e.name):
        if entry.is_file():
            dest = os.path.join(DATA_DIR, entry.name)
            shutil.copy2(entry.path, dest)
            print(f"  {entry.name} -> {dest}")
            copied += 1
    if copied == 0:
        print(f"  Warning: no files found in {src_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch dataset files into data_files/")
    parser.add_argument(
        "--local", metavar="PATH", default=None,
        help="Copy from a local directory instead of Google Drive",
    )
    args = parser.parse_args()

    if args.local:
        print(f"Fetching from local path: {args.local}")
        fetch_from_local(args.local)
    else:
        print(f"Fetching from Google Drive: {GDRIVE_FOLDER_URL}")
        fetch_from_drive()

    print("Done.")
