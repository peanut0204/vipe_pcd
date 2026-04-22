import os
from pathlib import Path
import csv

def create_video_csv(data_root, output_csv):
    """
    自動掃描目錄結構並生成CSV檔案
    
    Args:
        data_root: 360度影片的根目錄 (例如: "data/360-videos-new")
        output_csv: 輸出的CSV檔案名稱 (例如: "video_list.csv")
    """
    
    video_list = []
    data_path = Path(data_root)
    
    # 檢查根目錄是否存在
    if not data_path.exists():
        print(f"錯誤：目錄 {data_root} 不存在！")
        return
    
    print(f"開始掃描目錄：{data_root}")
    
    # 遍歷所有類別資料夾 (Autos_Vehicles, Comedy, Education...)
    for category_dir in data_path.iterdir():
        if not category_dir.is_dir():
            continue
            
        category_name = category_dir.name
        print(f"  處理類別：{category_name}")
        
        # 遍歷該類別下的所有影片ID資料夾
        for video_id_dir in category_dir.iterdir():
            if not video_id_dir.is_dir():
                continue
                
            video_id = video_id_dir.name
            print(f"    處理影片ID：{video_id}")
            
            # 尋找該影片ID資料夾下的所有.mp4檔案
            mp4_files = list(video_id_dir.glob("*.mp4"))
            
            if mp4_files:
                print(f"      找到 {len(mp4_files)} 個影片檔案")
                
                # 將每個.mp4檔案加入清單
                for mp4_file in sorted(mp4_files):
                    clip_filename = mp4_file.name
                    video_list.append([category_name, video_id, clip_filename])
            else:
                print(f"      警告：沒有找到.mp4檔案")
    
    # 寫入CSV檔案
    print(f"\n總共找到 {len(video_list)} 個影片檔案")
    print(f"正在寫入CSV檔案：{output_csv}")
    
    # 對整個清單進行最終排序：先按類別，再按影片ID，最後按檔案名稱數字
    def final_sort_key(row):
        category, video_id, filename = row
        try:
            # 嘗試將檔名轉成數字排序
            file_num = int(Path(filename).stem)
            return (category, video_id, file_num)
        except ValueError:
            # 如果檔名不是數字，用字串排序
            return (category, video_id, float('inf'), filename)
    
    video_list.sort(key=final_sort_key)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 可選：寫入標題行
        # writer.writerow(['category', 'video_id', 'clip_filename'])
        
        # 寫入所有影片資料
        for row in video_list:
            writer.writerow(row)
    
    print(f"CSV檔案建立完成！")
    
    # 顯示前幾行作為範例
    print(f"\n前5行內容預覽：")
    for i, row in enumerate(video_list[:5]):
        print(f"  {i+1}: {','.join(row)}")
    
    if len(video_list) > 5:
        print(f"  ... (還有 {len(video_list)-5} 行)")

def main():
    # 設定路徑
    data_root = "data/360-videos-new"  # 你的影片根目錄
    output_csv = "video_list.csv"      # 輸出的CSV檔案名稱
    
    # 建立CSV
    create_video_csv(data_root, output_csv)

if __name__ == "__main__":
    main()