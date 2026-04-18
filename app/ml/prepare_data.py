from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def list_videos(raw_dir: Path) -> list[Path]:
    return sorted([p for p in raw_dir.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".mkv"}])


def split_paths(paths: list[Path], train_ratio: float, val_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
    n = len(paths)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return paths[:train_end], paths[train_end:val_end], paths[val_end:]


def write_manifest(items: Iterable[Path], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(str(p) for p in items))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits by session files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    videos = list_videos(args.raw_dir)
    train, val, test = split_paths(videos, args.train_ratio, args.val_ratio)

    write_manifest(train, args.split_dir / "train.txt")
    write_manifest(val, args.split_dir / "val.txt")
    write_manifest(test, args.split_dir / "test.txt")

    print(f"Created splits from {len(videos)} videos.")
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
