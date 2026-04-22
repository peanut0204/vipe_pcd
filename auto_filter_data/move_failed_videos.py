import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="将所有失败的视频移动到新文件夹并统计剩余视频")
    parser.add_argument(
        "--failed_csv_dir",
        type=str,
        required=True,
        help="包含所有失败CSV文件的目录"
    )
    parser.add_argument(
        "--source_root",
        type=str,
        required=True,
        help="源视频文件的根目录"
    )
    parser.add_argument(
        "--target_root",
        type=str,
        required=True,
        help="目标根目录（失败视频存放位置）"
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
        help="是否保持原始的目录结构"
    )
    parser.add_argument(
        "--csv_prefix",
        type=str,
        default="",
        help="CSV文件的前缀（如果有的话）"
    )
    return parser.parse_args()

def count_videos_in_directory(directory, extension=".mp4"):
    """统计目录中的视频文件数量"""
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    pattern = f"**/*{extension}"
    video_files = list(directory.glob(pattern))
    return len(video_files)

def collect_all_failed_videos(csv_dir, csv_prefix=""):
    """从所有失败的CSV文件中收集视频信息"""
    csv_dir = Path(csv_dir)
    
    # 查找所有失败的CSV文件
    failed_csv_pattern = f"{csv_prefix}*_failed.csv" if csv_prefix else "*_failed.csv"
    csv_files = list(csv_dir.glob(failed_csv_pattern))
    
    print(f"找到 {len(csv_files)} 个失败的CSV文件:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    all_failed_videos = []
    failure_stats = {}
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            failure_type = csv_file.stem  # 文件名去掉扩展名
            failure_stats[failure_type] = len(df)
            
            # 添加失败类型信息
            df['failure_type'] = failure_type
            all_failed_videos.append(df)
            
            print(f"  {failure_type}: {len(df)} 个视频")
        except Exception as e:
            print(f"  错误读取 {csv_file}: {e}")
    
    if all_failed_videos:
        combined_df = pd.concat(all_failed_videos, ignore_index=True)
        # 去重（一个视频可能在多个失败类型中出现）
        unique_failed = combined_df.drop_duplicates(subset=['category', 'video_id', 'clip_id'])
        print(f"\n总失败视频数（去重后）: {len(unique_failed)}")
        return unique_failed, failure_stats
    else:
        return pd.DataFrame(), {}

def main():
    args = parse_args()
    
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    
    # 创建目标目录
    target_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("移动失败视频脚本")
    print("=" * 60)
    
    # 统计移动前的视频数量
    print("\n1. 统计移动前的视频数量...")
    original_count = count_videos_in_directory(source_root, args.video_extension)
    print(f"源目录中的视频总数: {original_count}")
    
    # 收集所有失败的视频
    print("\n2. 收集所有失败的视频信息...")
    failed_videos_df, failure_stats = collect_all_failed_videos(args.failed_csv_dir, args.csv_prefix)
    
    if failed_videos_df.empty:
        print("没有找到失败的视频，程序结束。")
        return
    
    # 开始移动视频
    print(f"\n3. 开始移动 {len(failed_videos_df)} 个失败的视频...")
    
    moved_count = 0
    failed_count = 0
    not_found_count = 0
    move_log = []
    
    for idx, row in tqdm(failed_videos_df.iterrows(), total=len(failed_videos_df)):
        category = row['category']
        video_id = row['video_id']
        clip_id = row['clip_id']
        failure_type = row.get('failure_type', 'unknown')
        
        # 构建源文件路径
        source_file = source_root / category / video_id / f"{clip_id}{args.video_extension}"
        
        if not source_file.exists():
            not_found_count += 1
            move_log.append(f"✗ 源文件不存在: {source_file}")
            continue
        
        # 构建目标文件路径
        if args.keep_structure:
            # 保持目录结构，但在失败类型下分类
            target_subdir = target_root / failure_type / category / video_id
            target_subdir.mkdir(parents=True, exist_ok=True)
            target_file = target_subdir / f"{clip_id}{args.video_extension}"
        else:
            # 按失败类型分类，文件名包含完整信息
            target_subdir = target_root / failure_type
            target_subdir.mkdir(parents=True, exist_ok=True)
            target_filename = f"{category}_{video_id}_{clip_id}{args.video_extension}"
            target_file = target_subdir / target_filename
        
        try:
            # 移动文件
            shutil.move(str(source_file), str(target_file))
            moved_count += 1
            move_log.append(f"✓ {source_file} -> {target_file}")
        except Exception as e:
            failed_count += 1
            move_log.append(f"✗ 移动失败 {source_file}: {e}")
    
    # 统计移动后的视频数量
    print(f"\n4. 统计移动后的视频数量...")
    remaining_count = count_videos_in_directory(source_root, args.video_extension)
    moved_total = original_count - remaining_count
    
    # 生成移动报告
    print(f"\n" + "=" * 60)
    print("移动完成报告")
    print("=" * 60)
    print(f"原始视频总数: {original_count}")
    print(f"尝试移动的失败视频数: {len(failed_videos_df)}")
    print(f"成功移动: {moved_count}")
    print(f"移动失败: {failed_count}")
    print(f"文件不存在: {not_found_count}")
    print(f"实际移动的视频数: {moved_total}")
    print(f"源目录剩余视频数: {remaining_count}")
    print(f"移动比例: {moved_total/original_count*100:.2f}%" if original_count > 0 else "0%")
    
    print(f"\n各失败类型统计:")
    for failure_type, count in failure_stats.items():
        print(f"  {failure_type}: {count} 个视频")
    
    # 保存详细日志
    log_file = target_root / "move_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("失败视频移动日志\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"源目录: {source_root}\n")
        f.write(f"目标目录: {target_root}\n")
        f.write(f"原始视频总数: {original_count}\n")
        f.write(f"成功移动: {moved_count}\n")
        f.write(f"移动失败: {failed_count}\n")
        f.write(f"文件不存在: {not_found_count}\n")
        f.write(f"实际移动数: {moved_total}\n")
        f.write(f"剩余视频数: {remaining_count}\n\n")
        
        f.write("各失败类型统计:\n")
        for failure_type, count in failure_stats.items():
            f.write(f"  {failure_type}: {count} 个视频\n")
        f.write("\n")
        
        f.write("详细移动日志:\n")
        f.write("-" * 30 + "\n")
        for log_entry in move_log:
            f.write(f"{log_entry}\n")
    
    # 保存移动后的统计信息
    stats_file = target_root / "move_statistics.csv"
    stats_data = {
        'metric': [
            'original_count', 'moved_count', 'failed_count', 
            'not_found_count', 'remaining_count', 'actual_moved'
        ],
        'value': [
            original_count, moved_count, failed_count,
            not_found_count, remaining_count, moved_total
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_file, index=False)
    
    print(f"\n详细日志已保存到: {log_file}")
    print(f"统计信息已保存到: {stats_file}")
    print(f"\n目标目录结构: {target_root}")

if __name__ == "__main__":
    main()