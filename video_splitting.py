# import cv2
# import os

# def split_video_with_info(input_folder, output_folder, frame_count=93):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
#     # 取得資料夾內所有影片檔
#     video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
#     video_files = video_files[:10] # split 10 videos first ############################
    
#     print(f"找到 {len(video_files)} 部影片，準備開始處理...\n")

#     for filename in video_files:
#         video_path = os.path.join(input_folder, filename)
#         cap = cv2.VideoCapture(video_path)
        
#         # 獲取影片資訊
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         # --- 這裡會先列印資訊 ---
#         print(f"🎬 檔名: {filename}")
#         print(f"   - 總幀數 (Total Frames): {total_frames}")
#         print(f"   - FPS: {fps:.2f}")
#         print(f"   - 預計切出段數: {total_frames // frame_count}")
#         print("-" * 30)

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         segment_idx = 0
#         frames_buffer = []

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frames_buffer.append(frame)

#             if len(frames_buffer) == frame_count:
#                 output_name = f"{os.path.splitext(filename)[0]}_part{segment_idx:03d}.mp4"
#                 output_path = os.path.join(output_folder, output_name)
                
#                 out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#                 for f in frames_buffer:
#                     out.write(f)
#                 out.release()
                
#                 segment_idx += 1
#                 frames_buffer = []

#         cap.release()
#         print(f"✅ {filename} 處理完成。\n")

# # --- 設定路徑 ---
# input_dir = './360-videos'  
# output_dir = './splitted_360-videos' 

# split_video_with_info(input_dir, output_dir)


import cv2
import os

def initialize_processed_log(input_folder, output_folder, log_file="processed_log.txt"):
    """
    掃描輸出資料夾，根據已存在的檔名回推哪些原始影片已處理，並更新紀錄檔。
    """
    print("🔍 正在初始化處理紀錄...")
    processed_videos = set()
    
    # 如果紀錄檔已存在，先讀取現有的
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            processed_videos.update([line.strip() for line in f.readlines()])

    # 掃描輸出資料夾中的檔案 (假設格式為 原檔名_partXXX.ext)
    if os.path.exists(output_folder):
        for f in os.listdir(output_folder):
            if "_part" in f:
                # 取得 _part 之前的原始檔名
                original_name_base = f.split('_part')[0]
                processed_videos.add(original_name_base)

    # 檢查輸入資料夾，找出完整的原始檔名並寫入紀錄
    final_list = []
    all_inputs = os.listdir(input_folder)
    for video in all_inputs:
        base_name = os.path.splitext(video)[0]
        if base_name in processed_videos or video in processed_videos:
            final_list.append(video)

    with open(log_file, 'w', encoding='utf-8') as f:
        for name in sorted(set(final_list)):
            f.write(f"{name}\n")
    
    print(f"✅ 初始化完成，目前紀錄中有 {len(final_list)} 部影片已處理過。\n")


def split_video_with_info(input_folder, output_folder, frame_count=93, log_file="processed_log.txt"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 支援格式增加 .webm
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    
    # 2. 讀取紀錄檔
    processed_list = []
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            processed_list = [line.strip() for line in f.readlines()]

    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions) and f not in processed_list]
    
    if not video_files:
        print("🙌 所有影片皆已處理完畢，沒有新任務。")
        return

    print(f"🚀 準備處理 {len(video_files)} 部新影片...\n")

    for filename in video_files:
        video_path = os.path.join(input_folder, filename)
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # WebM 判斷與 FourCC 設定
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.webm':
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        segment_idx = 0
        frames_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_buffer.append(frame)

            if len(frames_buffer) == frame_count:
                output_name = f"{os.path.splitext(filename)[0]}_part{segment_idx:03d}{ext}"
                out = cv2.VideoWriter(os.path.join(output_folder, output_name), fourcc, fps, (width, height))
                for f in frames_buffer:
                    out.write(f)
                out.release()
                segment_idx += 1
                frames_buffer = []

        cap.release()
        
        # 處理完一部就紀錄一部，防止中途當機
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(filename + '\n')
        print(f"✨ 已完成: {filename}")

# --- 執行流程 ---
input_dir = './360-videos'  
output_dir = './splitted_360-videos' 
log_path = 'processed_log.txt'

# 先執行初始化紀錄
initialize_processed_log(input_dir, output_dir, log_path)

# 再開始切片任務
split_video_with_info(input_dir, output_dir, log_file=log_path)