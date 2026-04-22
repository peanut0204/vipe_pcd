import os
import shutil
import random
from pathlib import Path

def create_validation_set():
    # 源路径和目标路径
    source_path = Path("data/360-videos-train")
    target_path = Path("data/360-videos-val")
    
    # 创建目标文件夹
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 所有类别
    categories = [
        "Autos_Vehicles",
        "Comedy", 
        "Education",
        "Entertainment",
        "Film_Animation",
        "Gaming",
        "Howto_Style",
        "News_Politics",
        "Nonprofits_Activism",
        "People_Blogs",
        "Pets_Animals",
        "Science_Technology",
        "Sports"
    ]
    
    # 设置随机种子确保可重现性
    random.seed(42)
    
    moved_folders = {}
    
    for category in categories:
        category_source = source_path / category
        category_target = target_path / category
        
        if not category_source.exists():
            print(f"警告: 类别 {category} 不存在于源路径中")
            continue
            
        # 获取该类别下的所有文件夹
        video_folders = [f for f in category_source.iterdir() if f.is_dir()]
        
        if len(video_folders) < 2:
            print(f"警告: 类别 {category} 只有 {len(video_folders)} 个文件夹，无法选择2个")
            continue
        
        # 随机选择2个文件夹
        selected_folders = random.sample(video_folders, 2)
        moved_folders[category] = [f.name for f in selected_folders]
        
        # 创建目标类别文件夹
        category_target.mkdir(parents=True, exist_ok=True)
        
        # 移动选中的文件夹
        for folder in selected_folders:
            source_folder = category_source / folder.name
            target_folder = category_target / folder.name
            
            try:
                # 移动整个文件夹（包含所有mp4文件）
                shutil.move(str(source_folder), str(target_folder))
                print(f"已移动: {category}/{folder.name}")
            except Exception as e:
                print(f"移动失败 {category}/{folder.name}: {e}")
    
    # 打印移动的文件夹汇总
    print("\n=== 移动汇总 ===")
    total_moved = 0
    for category, folders in moved_folders.items():
        print(f"{category}: {folders}")
        total_moved += len(folders)
    
    print(f"\n总共移动了 {total_moved} 个视频文件夹")
    print(f"验证集创建完成，路径: {target_path}")
    
    return moved_folders

if __name__ == "__main__":
    # 确认操作
    print("这将从训练集中移动每个类别的2个文件夹到验证集")
    print("源路径: data/360-videos-train")
    print("目标路径: data/360-videos-val")
    
    confirm = input("确认执行吗？(y/N): ")
    if confirm.lower() == 'y':
        moved_folders = create_validation_set()
    else:
        print("操作取消")