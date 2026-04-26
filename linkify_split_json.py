#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from typing import Any


def _is_probably_path(value: str) -> bool:
    return "/" in value or value.startswith(".")


def _linkify(files: list[str], base_dir: str) -> tuple[list[str], int]:
    linked: list[str] = []
    changed = 0
    for item in files:
        if not isinstance(item, str):
            linked.append(item)
            continue

        if os.path.isabs(item):
            linked.append(item)
            continue

        if _is_probably_path(item):
            normalized = os.path.normpath(os.path.join(base_dir, item))
        else:
            normalized = os.path.join(base_dir, item)

        linked.append(normalized)
        if normalized != item:
            changed += 1
    return linked, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a split-video JSON manifest so each entry in 'files' "
            "is an absolute path under a target directory."
        )
    )
    parser.add_argument("json_path", help="Path to the .json manifest to rewrite")
    parser.add_argument(
        "--target-dir",
        default="/workspace/vipe_pcd/splitted_360-videos",
        help="Directory to prefix onto each filename (default: %(default)s)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the manifest in-place (creates a .bak backup)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: <json_path>.linked.json when not using --in-place)",
    )
    parser.add_argument(
        "--verify-exists",
        action="store_true",
        help="Count how many linked files exist on disk (does not fail)",
    )

    args = parser.parse_args()
    json_path = args.json_path
    target_dir = os.path.abspath(args.target_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    files = data.get("files")
    if not isinstance(files, list):
        raise SystemExit("Expected JSON key 'files' to be a list")

    linked_files, changed = _linkify(files, target_dir)
    data["files"] = linked_files
    if isinstance(data.get("directory"), str):
        data["directory"] = target_dir

    if args.in_place:
        out_path = json_path
        backup_path = json_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(json_path, backup_path)
    else:
        out_path = args.out or (json_path + ".linked.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    exists_count = None
    if args.verify_exists:
        exists_count = sum(1 for p in linked_files if isinstance(p, str) and os.path.exists(p))

    print(f"wrote: {out_path}")
    print(f"target_dir: {target_dir}")
    print(f"entries: {len(linked_files)}")
    print(f"changed: {changed}")
    if exists_count is not None:
        print(f"exists: {exists_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
