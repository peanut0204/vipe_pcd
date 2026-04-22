#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm

def is_main_video(p: Path) -> bool:
    name = p.name
    return (
        p.suffix.lower() == ".mp4"
        and not name.endswith("_pcd.mp4")
        and not name.endswith("_mask.mp4")
    )

def fast_frame_count_at_least(path: Path, threshold: int) -> bool:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fc > 0:
        cap.release()
        return fc >= threshold
    cnt = 0
    ok = True
    while cnt < threshold:
        ok, _ = cap.read()
        if not ok:
            break
        cnt += 1
    cap.release()
    return cnt >= threshold

def ensure_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

def main():
    ap = argparse.ArgumentParser(description="找出缺件或影格不足的影片組，並保留結構地移動到目的資料夾。")
    ap.add_argument("src_root", type=Path, help="來源根目錄")
    ap.add_argument("dst_root", type=Path, help="目的根目錄（會建立相同結構）")
    ap.add_argument("--min_frames", type=int, default=25, help="pcd/mask 至少需要的影格數")
    ap.add_argument("--dry_run", action="store_true", help="只列印將要移動的檔案，不實際移動")
    args = ap.parse_args()

    src_root: Path = args.src_root.resolve()
    dst_root: Path = args.dst_root.resolve()

    if not src_root.exists():
        raise SystemExit(f"來源不存在：{src_root}")

    # 找出所有 main videos
    all_main_videos = []
    for dirpath in sorted(src_root.rglob("*")):
        if dirpath.is_dir():
            mp4s = sorted([p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
            main_videos = [p for p in mp4s if is_main_video(p)]
            all_main_videos.extend(main_videos)

    selected_groups = 0
    moved_files = 0

    # 用 tqdm 顯示進度
    for main_vid in tqdm(all_main_videos, desc="Processing videos", unit="video"):
        stem = main_vid.with_suffix("").name
        pcd = main_vid.with_name(f"{stem}_pcd.mp4")
        mask = main_vid.with_name(f"{stem}_mask.mp4")

        missing = (not pcd.exists()) or (not mask.exists())
        too_short = False
        if pcd.exists() and not fast_frame_count_at_least(pcd, args.min_frames):
            too_short = True
        if mask.exists() and not fast_frame_count_at_least(mask, args.min_frames):
            too_short = True

        if missing or too_short:
            selected_groups += 1
            reason = []
            if missing:
                miss_list = []
                if not pcd.exists(): miss_list.append("pcd缺少")
                if not mask.exists(): miss_list.append("mask缺少")
                reason.append("、".join(miss_list))
            if too_short:
                reason.append(f"影格< {args.min_frames}")
            reason_str = " & ".join(reason)

            rel_dir = main_vid.parent.relative_to(src_root)
            dst_dir = dst_root / rel_dir

            files_to_move = [main_vid]
            if pcd.exists(): files_to_move.append(pcd)
            if mask.exists(): files_to_move.append(mask)

            print(f"\n[選中] {main_vid}  ({reason_str})")
            for src_file in files_to_move:
                dst_file = dst_dir / src_file.name
                print(f"  -> {dst_file}")
                if not args.dry_run:
                    ensure_move(src_file, dst_file)
                    moved_files += 1

    print(f"\n完成。選中 {selected_groups} 組影片；移動檔案數：{moved_files}（dry-run={args.dry_run}）。")
    print(f"來源：{src_root}\n目的：{dst_root}")

if __name__ == "__main__":
    main()
