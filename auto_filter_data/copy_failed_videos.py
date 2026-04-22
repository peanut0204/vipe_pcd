import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="从失败的CSV文件中复制前N个视频到指定目录")
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="失败视频的CSV文件路径 (例如: optical_flow_failed.csv)"
    )
    parser.add_argument(
        "--source_root",
        type=str,
        required=True,
        help="源视频文件的根目录"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="目标文件夹路径"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="要复制的视频数量 (默认: 100)"
    )
    parser.add_argument(
        "--video_extension",
        type=str,
        default=".mp4",
        help="视频文件扩展名 (默认: .mp4)"
    )
    parser.add_argument(
        "--keep_structure",
        action="store_true",
        help="是否保持原始的目录结构 (category/video_id/)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 读取CSV文件
    try:
        df = pd.read_csv(args.csv_file)
        print(f"成功读取CSV文件: {args.csv_file}")
        print(f"总共有 {len(df)} 个失败视频")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 取前N个
    df_subset = df.head(args.count)
    print(f"将复制前 {len(df_subset)} 个视频")
    
    # 创建目标目录
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    source_root = Path(args.source_root)
    
    copied_count = 0
    failed_count = 0
    copy_log = []
    
    print("\n开始复制视频文件...")
    
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        category = row['category']
        video_id = row['video_id']
        clip_id = row['clip_id']
        
        # 构建源文件路径
        source_file = source_root / category / video_id / f"{clip_id}{args.video_extension}"
        
        if source_file.exists():
            if args.keep_structure:
                # 保持目录结构
                target_subdir = target_dir / category / video_id
                target_subdir.mkdir(parents=True, exist_ok=True)
                target_file = target_subdir / f"{clip_id}{args.video_extension}"
            else:
                # 扁平化存储，文件名包含分类和视频ID信息
                target_filename = f"{category}_{video_id}_{clip_id}{args.video_extension}"
                target_file = target_dir / target_filename
            
            try:
                shutil.copy2(source_file, target_file)
                copied_count += 1
                copy_log.append(f"✓ {source_file} -> {target_file}")
            except Exception as e:
                failed_count += 1
                copy_log.append(f"✗ 复制失败 {source_file}: {e}")
        else:
            failed_count += 1
            copy_log.append(f"✗ 源文件不存在: {source_file}")
    
    # 保存复制日志
    log_file = target_dir / "copy_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"视频复制日志\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"源CSV文件: {args.csv_file}\n")
        f.write(f"源目录: {args.source_root}\n")
        f.write(f"目标目录: {args.target_dir}\n")
        f.write(f"请求复制数量: {args.count}\n")
        f.write(f"成功复制: {copied_count}\n")
        f.write(f"失败: {failed_count}\n\n")
        f.write("详细日志:\n")
        f.write("-" * 30 + "\n")
        for log_entry in copy_log:
            f.write(f"{log_entry}\n")
    
    # 保存复制的视频列表
    if copied_count > 0:
        copied_videos = df_subset.head(copied_count).copy()
        copied_list_file = target_dir / "copied_videos.csv"
        copied_videos.to_csv(copied_list_file, index=False)
        print(f"已保存复制的视频列表到: {copied_list_file}")
    
    print(f"\n复制完成!")
    print(f"成功复制: {copied_count} 个视频")
    print(f"失败: {failed_count} 个")
    print(f"目标目录: {target_dir}")
    print(f"详细日志: {log_file}")

if __name__ == "__main__":
    main()
    main()