#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Stats:
    total: int = 0
    created: int = 0
    skipped_existing: int = 0
    missing_source: int = 0
    conflicts: int = 0
    errors: int = 0


def _load_manifest(manifest_path: str) -> tuple[list[str], str | None]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    files = data.get("files")
    if not isinstance(files, list) or not all(isinstance(x, str) for x in files):
        raise SystemExit("Expected manifest JSON key 'files' to be a list[str].")
    directory = data.get("directory")
    if directory is not None and not isinstance(directory, str):
        directory = None
    return files, directory


def _resolve_source(path_entry: str, manifest_dir: str | None) -> str:
    if os.path.isabs(path_entry):
        return path_entry
    if manifest_dir:
        return os.path.normpath(os.path.join(manifest_dir, path_entry))
    return os.path.abspath(path_entry)


def _safe_relpath(target: str, start: str) -> str:
    try:
        return os.path.relpath(target, start)
    except ValueError:
        return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create symlinks for each entry in a JSON manifest's 'files' list. "
            "Links are created in the destination directory, named by basename."
        )
    )
    parser.add_argument("manifest", help="Path to the manifest JSON")
    parser.add_argument(
        "--dest",
        default="/workspace/vipe_pcd/selected_splitted_360_clips",
        help="Destination directory to place symlinks (default: %(default)s)",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Create relative symlinks (recommended when dest and sources are on same filesystem)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip entries whose source does not exist (default: create potentially-dangling symlinks)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing symlinks (will not overwrite regular files)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N entries (for quick tests)",
    )

    args = parser.parse_args()
    dest_dir = os.path.abspath(args.dest)
    os.makedirs(dest_dir, exist_ok=True)

    files, manifest_dir = _load_manifest(args.manifest)
    if args.limit is not None:
        files = files[: max(args.limit, 0)]

    stats = Stats(total=len(files))
    for entry in files:
        try:
            source = _resolve_source(entry, manifest_dir)
            if not os.path.exists(source):
                stats.missing_source += 1
                if args.skip_missing:
                    continue

            link_name = os.path.basename(source)
            dest_path = os.path.join(dest_dir, link_name)

            if os.path.lexists(dest_path):
                if os.path.islink(dest_path):
                    if args.overwrite:
                        os.unlink(dest_path)
                    else:
                        stats.skipped_existing += 1
                        continue
                else:
                    stats.conflicts += 1
                    continue

            target = _safe_relpath(source, dest_dir) if args.relative else source
            os.symlink(target, dest_path)
            stats.created += 1
        except Exception:
            stats.errors += 1

    print(f"dest: {dest_dir}")
    print(f"entries: {stats.total}")
    print(f"created: {stats.created}")
    print(f"skipped_existing: {stats.skipped_existing}")
    print(f"missing_source: {stats.missing_source}")
    print(f"conflicts: {stats.conflicts}")
    print(f"errors: {stats.errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
