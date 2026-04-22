#!/usr/bin/env python3
"""
準備採樣訓練數據集 - 多進程版本 (修改版)
只處理3840x1920分辨率的視頻，從原始視頻中採樣49幀片段，每幀resize到1024x512，按目錄結構組織
"""

import json
import os
import glob
import subprocess
import time
import cv2
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, Manager
import functools

def get_video_resolution(video_file):
    """獲取視頻分辨率"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            video_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        width, height = map(int, result.stdout.strip().split(','))
        return width, height, True
    except:
        # 使用cv2備用方法
        try:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                return 0, 0, False
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return width, height, True
        except:
            return 0, 0, False

def run_ffmpeg_command(cmd):
    """執行 FFmpeg 命令"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"FFmpeg error: {e.stderr}"
    except Exception as e:
        return False, f"Command error: {str(e)}"

def check_video_health_with_cv2(video_file):
    """使用 cv2 檢查視頻文件是否健康可讀（與 dataloader 一致）"""
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return False, "cannot_open", 0, 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 檢查分辨率是否為3840x1920
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width != 3840 or height != 1920:
            cap.release()
            return False, f"wrong_resolution_{width}x{height}", 0, 0
        
        # 嘗試讀取第一幀
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, "cannot_read_frame", 0, 0
        
        # 嘗試讀取中間的幾幀
        if total_frames > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                return False, "cannot_read_middle_frame", 0, 0
        
        cap.release()
        return True, "healthy", total_frames, fps
        
    except Exception as e:
        return False, f"cv2_error_{str(e)}", 0, 0

def check_video_health(video_file):
    """檢查視頻文件是否健康可讀且分辨率正確"""
    # 首先使用 cv2 檢查（與 dataloader 一致）
    cv2_healthy, cv2_status, total_frames, fps = check_video_health_with_cv2(video_file)
    
    if cv2_healthy:
        # 如果 cv2 通過，再用 ffprobe 獲取編碼信息
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,width,height',
                '-of', 'csv=p=0',
                video_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
            codec_info = result.stdout.strip().split(',')
            codec = codec_info[0] if len(codec_info) > 0 else "unknown_codec"
            
            # 再次確認分辨率
            if len(codec_info) >= 3:
                width, height = int(codec_info[1]), int(codec_info[2])
                if width != 3840 or height != 1920:
                    return False, f"wrong_resolution_{width}x{height}", 0, 0
            
            return True, codec, total_frames, fps
        except:
            return True, "unknown_codec", total_frames, fps
    else:
        return False, cv2_status, 0, 0

def tensor_to_video(video_tensor, output_path, frame_rate=25.0):
    """
    高兼容性版本，使用 ffmpeg 后处理
    """
    import subprocess
    
    # 先保存为临时的 avi 文件
    temp_path = output_path.replace('.mp4', '_temp.avi')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_np = ((video_tensor + 1) * 127.5).clamp(0, 255).byte().numpy()
    video_np = video_np.transpose(0, 2, 3, 1)

    T, H, W, C = video_np.shape
    
    # 使用 XVID 编码器先保存为 AVI
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_path, fourcc, frame_rate, (W, H))
    
    if not out.isOpened():
        raise RuntimeError(f"无法创建临时视频文件: {temp_path}")
    
    for i in range(T):
        frame = video_np[i]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    # 使用 ffmpeg 转换为高兼容性的 MP4
    try:
        subprocess.run([
            'ffmpeg', '-i', temp_path, 
            '-c:v', 'libx264',  # H.264 编码
            '-preset', 'medium',
            '-crf', '23',       # 质量设置
            '-pix_fmt', 'yuv420p',  # 像素格式，兼容性好
            '-y',               # 覆盖输出文件
            output_path
        ], check=True, capture_output=True)
        
        # 删除临时文件
        os.remove(temp_path)
        print(f"保存完成: {output_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 转换失败: {e}")
        # 如果 ffmpeg 失败，至少保留 AVI 文件
        os.rename(temp_path, output_path.replace('.mp4', '.avi'))
        print(f"已保存为 AVI 格式: {output_path.replace('.mp4', '.avi')}")
    except FileNotFoundError:
        print("未找到 ffmpeg，已保存为 AVI 格式")
        os.rename(temp_path, output_path.replace('.mp4', '.avi'))
        print(f"已保存为 AVI 格式: {output_path.replace('.mp4', '.avi')}")

def sample_and_process_frames(input_file, start_frame, end_frame, target_fps=5.0, num_frames=25):
    """
    從視頻中採樣25幀，同時保存原始(3840x1920)和resize(1024x512)版本
    
    Args:
        input_file: 輸入視頻文件路徑
        start_frame: 開始幀
        end_frame: 結束幀
        target_fps: 目標fps (用於計算採樣間隔)
        num_frames: 採樣幀數 (25)
    
    Returns:
        high_res_frames: numpy array (25, 1920, 3840, 3) BGR格式 - 原始分辨率
        low_res_frames: numpy array (25, 512, 1024, 3) BGR格式 - resize後
        actual_fps: 實際fps
        success: 是否成功
    """
    try:
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            return None, None, 0, False
        
        # 檢查分辨率是否為3840x1920
        width_check = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_check = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width_check != 3840 or height_check != 1920:
            cap.release()
            return None, None, 0, False
        
        # 獲取視頻信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 計算採樣間隔
        frame_interval = max(round(original_fps / target_fps), 1)
        actual_fps = original_fps / frame_interval
        
        # 計算可用幀範圍
        clip_total_frames = end_frame - start_frame + 1
        required_frames = num_frames * frame_interval
        
        if required_frames > clip_total_frames:
            cap.release()
            return None, None, 0, False
        
        # 隨機選擇起始位置
        max_start_offset = clip_total_frames - required_frames
        start_offset = random.randint(0, max(0, max_start_offset))
        actual_start_frame = start_frame + start_offset
        
        # 採樣幀
        high_res_frames = []
        low_res_frames = []
        
        for i in range(num_frames):
            frame_idx = actual_start_frame + i * frame_interval
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                cap.release()
                return None, None, 0, False
            
            # 保存原始分辨率幀
            high_res_frames.append(frame.copy())
            
            # Resize到1024x512
            resized_frame = cv2.resize(frame, (1024, 512))
            low_res_frames.append(resized_frame)
        
        cap.release()
        
        # 轉換為numpy array
        high_res_frames_array = np.stack(high_res_frames, axis=0)  # (25, 1920, 3840, 3)
        low_res_frames_array = np.stack(low_res_frames, axis=0)   # (25, 512, 1024, 3)
        
        return high_res_frames_array, low_res_frames_array, actual_fps, True
        
    except Exception as e:
        print(f"採樣處理錯誤: {e}")
        return None, None, 0, False

def frames_to_video(frames, output_path, fps=5.0):
    """
    將採樣的幀保存為視頻文件
    
    Args:
        frames: numpy array (49, H, W, 3) BGR格式
        output_path: 輸出路徑
        fps: 幀率
    """
    try:
        # 創建輸出目錄
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 先保存為臨時 AVI 文件
        temp_path = output_path.replace('.mp4', '_temp.avi')
        
        T, H, W, C = frames.shape
        
        # 使用 XVID 編碼器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
        
        if not out.isOpened():
            raise RuntimeError(f"無法創建臨時視頻文件: {temp_path}")
        
        for i in range(T):
            out.write(frames[i])
        
        out.release()
        
        # 使用 ffmpeg 轉換為高兼容性的 MP4
        try:
            subprocess.run([
                'ffmpeg', '-i', temp_path, 
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-y',
                output_path
            ], check=True, capture_output=True)
            
            # 刪除臨時文件
            os.remove(temp_path)
            return True, ""
            
        except subprocess.CalledProcessError as e:
            # 如果 ffmpeg 失敗，保留 AVI 文件
            avi_path = output_path.replace('.mp4', '.avi')
            os.rename(temp_path, avi_path)
            return True, f"已保存為 AVI 格式: {avi_path}"
            
        except FileNotFoundError:
            # 如果沒有 ffmpeg，保留 AVI 文件
            avi_path = output_path.replace('.mp4', '.avi')
            os.rename(temp_path, avi_path)
            return True, f"未找到 ffmpeg，已保存為 AVI 格式: {avi_path}"
            
    except Exception as e:
        return False, f"保存視頻錯誤: {str(e)}"

def get_video_file_path(video_dir, video_id, extensions):
    """找到視頻文件的完整路徑"""
    for ext in extensions:
        file_path = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(file_path):
            return file_path
    return None

def process_single_clip(clip_data):
    """處理單個片段的函數（用於多進程）"""
    clip, worker_id = clip_data
    
    video_id = clip['video_id']
    category = clip['category']
    clip_id = clip['clip_id']
    
    # 檢查輸出文件是否已存在（兩個版本都要檢查）
    high_res_exists = os.path.exists(clip['high_res_output_file'])
    low_res_exists = os.path.exists(clip['low_res_output_file'])
    
    if high_res_exists and low_res_exists:
        return {
            'status': 'skip',
            'worker_id': worker_id,
            'video_id': video_id,
            'category': category,
            'clip_id': clip_id,
            'message': '已存在(兩個版本)'
        }
    
    # 創建輸出目錄
    os.makedirs(clip['high_res_output_dir'], exist_ok=True)
    os.makedirs(clip['low_res_output_dir'], exist_ok=True)
    
    # 採樣和處理幀 (獲得兩個版本)
    high_res_frames, low_res_frames, actual_fps, success = sample_and_process_frames(
        clip['input_file'],
        clip['start_frame'],
        clip['end_frame'],
        target_fps=5.0,
        num_frames=25
    )
    
    if not success:
        return {
            'status': 'fail',
            'worker_id': worker_id,
            'video_id': video_id,
            'category': category,
            'clip_id': clip_id,
            'error': '採樣失敗',
            'codec_info': clip.get('codec', 'unknown')
        }
    
    # 保存高分辨率版本 (3840x1920)
    high_res_success = True
    high_res_message = ""
    if not high_res_exists:
        high_res_success, high_res_message = frames_to_video(
            high_res_frames, clip['high_res_output_file'], actual_fps
        )
    
    # 保存低分辨率版本 (1024x512)
    low_res_success = True
    low_res_message = ""
    if not low_res_exists:
        low_res_success, low_res_message = frames_to_video(
            low_res_frames, clip['low_res_output_file'], actual_fps
        )
    
    # 判斷整體成功狀態
    if high_res_success and low_res_success:
        messages = []
        if high_res_message:
            messages.append(f"High-res: {high_res_message}")
        if low_res_message:
            messages.append(f"Low-res: {low_res_message}")
        
        return {
            'status': 'success',
            'worker_id': worker_id,
            'video_id': video_id,
            'category': category,
            'clip_id': clip_id,
            'fps': actual_fps,
            'codec_info': clip.get('codec', 'unknown'),
            'message': ' | '.join(messages) if messages else '',
            'high_res_saved': not high_res_exists,
            'low_res_saved': not low_res_exists
        }
    else:
        errors = []
        if not high_res_success:
            errors.append(f"High-res失敗: {high_res_message}")
        if not low_res_success:
            errors.append(f"Low-res失敗: {low_res_message}")
        
        return {
            'status': 'fail',
            'worker_id': worker_id,
            'video_id': video_id,
            'category': category,
            'clip_id': clip_id,
            'error': ' | '.join(errors),
            'codec_info': clip.get('codec', 'unknown')
        }

def scan_existing_healthy_clips(reference_dir):
    """掃描參考目錄中的健康視頻片段（具體的片段文件）"""
    healthy_clips = set()
    
    if not os.path.exists(reference_dir):
        print(f"⚠️  參考目錄不存在: {reference_dir}")
        return healthy_clips
    
    print(f"🔍 掃描參考目錄中的健康片段: {reference_dir}")
    
    # 搜索所有 .mp4 和 .avi 文件
    for ext in ['*.mp4', '*.avi']:
        pattern = os.path.join(reference_dir, "**", ext)
        existing_files = glob.glob(pattern, recursive=True)
        
        for file_path in existing_files:
            # 解析路徑: reference_dir/category/video_id/clip_id.mp4
            rel_path = os.path.relpath(file_path, reference_dir)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) >= 3:
                category = path_parts[0]
                video_id = path_parts[1]
                clip_filename = path_parts[2]
                clip_id = os.path.splitext(clip_filename)[0]
                
                # 記錄具體的片段標識符
                healthy_clips.add(f"{category}/{video_id}/{clip_id}")
    
    print(f"✅ 找到 {len(healthy_clips)} 個參考健康片段")
    return healthy_clips
def scan_existing_clips(high_res_output_dir, low_res_output_dir):
    """掃描已存在的視頻片段（兩個目錄）"""
    existing_clips = set()
    
    # 掃描高分辨率目錄
    if os.path.exists(high_res_output_dir):
        print("🔍 掃描已存在的高分辨率視頻片段...")
        for ext in ['*.mp4', '*.avi']:
            pattern = os.path.join(high_res_output_dir, "**", ext)
            existing_files = glob.glob(pattern, recursive=True)
            
            for file_path in existing_files:
                rel_path = os.path.relpath(file_path, high_res_output_dir)
                path_parts = rel_path.split(os.sep)
                
                if len(path_parts) >= 3:
                    category = path_parts[0]
                    video_id = path_parts[1]
                    clip_filename = path_parts[2]
                    clip_id = os.path.splitext(clip_filename)[0]
                    existing_clips.add(f"{category}/{video_id}/{clip_id}")
    
    # 掃描低分辨率目錄
    if os.path.exists(low_res_output_dir):
        print("🔍 掃描已存在的低分辨率視頻片段...")
        for ext in ['*.mp4', '*.avi']:
            pattern = os.path.join(low_res_output_dir, "**", ext)
            existing_files = glob.glob(pattern, recursive=True)
            
            for file_path in existing_files:
                rel_path = os.path.relpath(file_path, low_res_output_dir)
                path_parts = rel_path.split(os.sep)
                
                if len(path_parts) >= 3:
                    category = path_parts[0]
                    video_id = path_parts[1]
                    clip_filename = path_parts[2]
                    clip_id = os.path.splitext(clip_filename)[0]
                    existing_clips.add(f"{category}/{video_id}/{clip_id}")
    
    print(f"✅ 找到 {len(existing_clips)} 個已存在的片段")
    return existing_clips

def prepare_sampled_dataset():
    # 設定路徑
    clips_info_file = "process_data/clips_info.jsonl"
    filtered_file = "process_data/clips_filtered_high_quality.txt"
    video_dir = "/home/ark/local/koi/360-1M/360-videos"
    reference_healthy_dir = "data/360-videos-new"  # 參考健康片段目錄
    
    # 兩個輸出目錄
    high_res_output_dir = "data/360-videos-3840x1920-25frames-high-res"  # 原始分辨率
    low_res_output_dir = "data/360-videos-3840x1920-25frames-low-res"   # resize後分辨率
    
    # 多進程設定
    NUM_WORKERS = 8
    
    # 常見的視頻檔案副檔名
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    
    print("=" * 80)
    print("🎯 準備採樣訓練數據集 - 3840x1920 分辨率，25幀，雙版本")
    print("   只處理3840x1920分辨率的視頻，每個clip採樣25幀")
    print("   同時保存高分辨率(3840x1920)和低分辨率(1024x512)版本")
    print("   只處理在參考目錄中存在的健康片段")
    print("=" * 80)
    print(f"📖 clips_info.jsonl: {clips_info_file}")
    print(f"📋 filtered list: {filtered_file}")
    print(f"📁 video directory: {video_dir}")
    print(f"🏥 reference healthy dir: {reference_healthy_dir}")
    print(f"💾 high-res output: {high_res_output_dir}")
    print(f"💾 low-res output: {low_res_output_dir}")
    print(f"👥 workers: {NUM_WORKERS}")
    print(f"🎬 采样参数: 25帧, 高分辨率(3840x1920) + 低分辨率(1024x512)")
    print()
    
    # 檢查依賴
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg 和 FFprobe 已安裝並可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 錯誤: FFmpeg 或 FFprobe 未安裝或不在 PATH 中")
        print("請安裝 FFmpeg: sudo apt install ffmpeg")
        return False
    
    try:
        import cv2
        import numpy as np
        print("✅ OpenCV 和 NumPy 已安裝並可用")
    except ImportError:
        print("❌ 錯誤: OpenCV 或 NumPy 未安裝")
        print("請安裝: pip install opencv-python numpy")
        return False
    print()
    
    # 檢查輸入文件和目錄
    if not os.path.exists(clips_info_file):
        print(f"❌ 錯誤: clips_info.jsonl 不存在: {clips_info_file}")
        return False
    
    if not os.path.exists(filtered_file):
        print(f"❌ 錯誤: filtered file 不存在: {filtered_file}")
        return False
    
    if not os.path.exists(video_dir):
        print(f"❌ 錯誤: video directory 不存在: {video_dir}")
        return False
    
    if not os.path.exists(reference_healthy_dir):
        print(f"❌ 錯誤: reference healthy directory 不存在: {reference_healthy_dir}")
        return False
    
    # 創建輸出目錄
    Path(high_res_output_dir).mkdir(exist_ok=True)
    Path(low_res_output_dir).mkdir(exist_ok=True)
    print(f"✅ 高分辨率輸出目錄已準備: {high_res_output_dir}")
    print(f"✅ 低分辨率輸出目錄已準備: {low_res_output_dir}")
    print()
    
    # Step 0: 掃描參考健康片段
    healthy_clips = scan_existing_healthy_clips(reference_healthy_dir)
    print()
    
    # Step 0.1: 掃描已存在的片段
    existing_clips = scan_existing_clips(high_res_output_dir, low_res_output_dir)
    print()
    
    # Step 1: 掃描實際存在的視頻文件並檢查分辨率
    print("🔍 Step 1: 掃描 360-videos 目錄並檢查分辨率...")
    existing_videos = set()
    valid_3840x1920_videos = set()  # 記錄3840x1920的視頻
    resolution_stats = defaultdict(int)
    file_count = 0
    
    for ext in video_extensions:
        pattern = os.path.join(video_dir, f"*{ext}")
        files = glob.glob(pattern)
        file_count += len(files)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            existing_videos.add(name_without_ext)
            
            # 檢查分辨率
            width, height, success = get_video_resolution(file_path)
            if success:
                resolution_key = f"{width}x{height}"
                resolution_stats[resolution_key] += 1
                
                # 記錄3840x1920的視頻
                if width == 3840 and height == 1920:
                    valid_3840x1920_videos.add(name_without_ext)
    
    print(f"✅ 找到 {file_count} 個視頻文件，{len(existing_videos)} 個唯一視頻")
    print(f"🎯 其中 {len(valid_3840x1920_videos)} 個是3840x1920分辨率")
    
    # 顯示分辨率統計
    print("\n📊 分辨率分布 (前10):")
    for resolution, count in sorted(resolution_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = count / len(existing_videos) * 100 if len(existing_videos) > 0 else 0
        mark = "🎯" if resolution == "3840x1920" else "  "
        print(f"{mark} {resolution:15s}: {count:4d} ({percentage:5.1f}%)")
    print()
    
    # Step 2: 讀取 clips_info.jsonl
    print("📖 Step 2: 讀取 clips_info.jsonl...")
    video_info = {}
    
    try:
        with open(clips_info_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        video_name = data.get('video_name')
                        if video_name:
                            video_info[video_name] = data
                    except json.JSONDecodeError as e:
                        print(f"⚠️  警告: 第 {line_num} 行 JSON 解析錯誤: {e}")
    except Exception as e:
        print(f"❌ 錯誤: 無法讀取 clips_info.jsonl: {e}")
        return False
    
    print(f"✅ 從 clips_info.jsonl 讀取 {len(video_info)} 個視頻的信息")
    print()
    
    # Step 3: 讀取高品質片段列表（只處理3840x1920且在參考目錄中的視頻）
    print("📋 Step 3: 讀取高品質片段列表（只處理3840x1920且健康的視頻）...")
    high_quality_clips = []
    category_stats = defaultdict(int)
    skipped_wrong_resolution = 0
    skipped_unhealthy = 0
    
    try:
        with open(filtered_file, 'r', encoding='utf-8') as f:
            # 跳過標題行
            header = f.readline().strip()
            print(f"📄 檔案標題: {header}")
            
            for line_num, line in enumerate(f, 2):
                line = line.strip()
                if line:
                    try:
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            category = parts[0]
                            video_id = parts[1]
                            clip_id = int(parts[2])
                            
                            # 檢查視頻是否為3840x1920
                            if video_id not in valid_3840x1920_videos:
                                skipped_wrong_resolution += 1
                                continue
                            
                            # 檢查片段是否在參考健康目錄中
                            clip_key = f"{category}/{video_id}/{clip_id}"
                            if clip_key not in healthy_clips:
                                skipped_unhealthy += 1
                                continue
                            
                            # 檢查是否已存在
                            if clip_key in existing_clips:
                                continue  # 跳過已存在的片段
                            
                            high_quality_clips.append({
                                'category': category,
                                'video_id': video_id,
                                'clip_id': clip_id
                            })
                            category_stats[category] += 1
                        else:
                            print(f"⚠️  警告: 第 {line_num} 行格式不正確")
                    except Exception as e:
                        print(f"⚠️  警告: 第 {line_num} 行處理錯誤: {e}")
    except Exception as e:
        print(f"❌ 錯誤: 無法讀取 filtered file: {e}")
        return False
    
    print(f"✅ 需要處理 {len(high_quality_clips)} 個新片段 (3840x1920 + 健康)")
    print(f"⚠️  跳過 {skipped_wrong_resolution} 個非3840x1920分辨率的片段")
    print(f"🏥 跳過 {skipped_unhealthy} 個不健康的片段")
    print("📊 各類別需處理片段數量:")
    for category, count in sorted(category_stats.items()):
        print(f"   {category:20s}: {count:4d}")
    print()
    
    if len(high_quality_clips) == 0:
        print("🎉 沒有3840x1920且健康的片段需要處理，或所有片段都已存在！")
        return True
    
    # Step 4: 批量檢查視頻健康狀態
    print("🔄 Step 4: 批量檢查視頻健康狀態...")
    valid_clips = []
    missing_videos = set()
    missing_info = set()
    invalid_clips = []
    corrupted_videos = set()
    wrong_resolution_videos = set()  # 新增：記錄分辨率不符的視頻
    
    # 按 video_id 分組以減少重複檢查
    videos_to_check = set()
    for clip_data in high_quality_clips:
        videos_to_check.add(clip_data['video_id'])
    
    # 檢查視頻健康狀態
    video_health_cache = {}
    
    print(f"需要檢查 {len(videos_to_check)} 個唯一視頻的健康狀態...")
    
    for i, video_id in enumerate(videos_to_check, 1):
        print(f"[{i:4d}/{len(videos_to_check)}] 檢查 {video_id}", end=" ... ")
        
        if video_id not in existing_videos:
            print("❌ 不存在")
            missing_videos.add(video_id)
            continue
        
        if video_id not in video_info:
            print("❌ 缺少信息")
            missing_info.add(video_id)
            continue
        
        video_file_path = get_video_file_path(video_dir, video_id, video_extensions)
        if not video_file_path:
            print("❌ 找不到文件")
            missing_videos.add(video_id)
            continue
        
        healthy, codec, total_frames_check, fps_check = check_video_health(video_file_path)
        
        if not healthy:
            if "wrong_resolution" in codec:
                print(f"❌ 分辨率不符 ({codec})")
                wrong_resolution_videos.add(video_id)
            else:
                print(f"❌ 損壞 ({codec})")
                corrupted_videos.add(video_id)
        else:
            print(f"✅ 正常 3840x1920 ({codec})")
            video_health_cache[video_id] = {
                'codec': codec,
                'total_frames': total_frames_check,
                'fps': fps_check,
                'file_path': video_file_path
            }
    
    print()
    
    # 生成有效片段列表
    for clip_data in high_quality_clips:
        video_id = clip_data['video_id']
        clip_id = clip_data['clip_id']
        category = clip_data['category']
        
        # 跳過有問題的視頻
        if video_id in missing_videos or video_id in missing_info or video_id in corrupted_videos or video_id in wrong_resolution_videos:
            continue
        
        if video_id not in video_health_cache:
            continue
        
        health_info = video_health_cache[video_id]
        info = video_info[video_id]
        clip_info_list = info.get('clip_info', [])
        
        # 檢查 clip_id 是否有效
        clip_found = False
        for clip_info_dict in clip_info_list:
            if str(clip_id) in clip_info_dict:
                start_frame, end_frame = clip_info_dict[str(clip_id)]
                
                # 檢查片段是否超出視頻長度
                if health_info['total_frames'] > 0 and end_frame > health_info['total_frames']:
                    invalid_clips.append(f"{video_id}:{clip_id} (超出長度)")
                    continue
                
                # 檢查片段是否足夠長（25幀需要的幀數）
                clip_length = end_frame - start_frame + 1
                fps = info.get('fps', 30.0)
                frame_interval = max(round(fps / 5.0), 1)
                actual_required = 25 * frame_interval  # 25幀所需的最少幀數
                
                if clip_length < actual_required:
                    invalid_clips.append(f"{video_id}:{clip_id} (片段太短，需要{actual_required}幀，只有{clip_length}幀)")
                    continue
                
                valid_clips.append({
                    'video_id': video_id,
                    'category': category,
                    'clip_id': clip_id,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'fps': info.get('fps', 30.0),
                    'codec': health_info['codec'],
                    'input_file': health_info['file_path'],
                    'high_res_output_dir': os.path.join(high_res_output_dir, category, video_id),
                    'low_res_output_dir': os.path.join(low_res_output_dir, category, video_id),
                    'high_res_output_file': os.path.join(high_res_output_dir, category, video_id, f"{clip_id}.mp4"),
                    'low_res_output_file': os.path.join(low_res_output_dir, category, video_id, f"{clip_id}.mp4")
                })
                clip_found = True
                break
        
        if not clip_found:
            invalid_clips.append(f"{video_id}:{clip_id} (找不到片段)")
    
    # 統計結果
    print(f"✅ 準備採樣的片段 (3840x1920 + 健康): {len(valid_clips)}")
    print(f"⚠️  缺少視頻文件: {len(missing_videos)}")
    print(f"⚠️  缺少視頻信息: {len(missing_info)}")
    print(f"❌ 損壞的視頻文件: {len(corrupted_videos)}")
    print(f"❌ 分辨率不符的視頻: {len(wrong_resolution_videos)}")
    print(f"⚠️  無效片段 ID: {len(invalid_clips)}")
    print()
    
    if len(valid_clips) == 0:
        print("❌ 沒有找到任何有效的3840x1920健康片段可以採樣！")
        return False
    
    # Step 5: 多進程採樣和處理視頻
    print(f"🎬 Step 5: 開始多進程採樣視頻片段（{NUM_WORKERS} workers）...")
    print(f"總共需要處理 {len(valid_clips)} 個3840x1920健康片段")
    print("採樣參數: 25幀, 高分辨率(3840x1920) + 低分辨率(1024x512)")
    print()
    
    start_time = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_clips = []
    
    # 為每個片段添加 worker_id
    clip_data_with_worker = [(clip, i % NUM_WORKERS) for i, clip in enumerate(valid_clips)]
    
    # 使用多進程池處理
    with Pool(processes=NUM_WORKERS) as pool:
        results = []
        
        # 提交所有任務
        for clip_data in clip_data_with_worker:
            result = pool.apply_async(process_single_clip, (clip_data,))
            results.append(result)
        
        # 收集結果並顯示進度
        processed = 0
        for result in results:
            try:
                res = result.get(timeout=300)  # 5 分鐘超時
                processed += 1
                
                status = res['status']
                worker_id = res['worker_id']
                video_id = res['video_id']
                category = res['category']
                clip_id = res['clip_id']
                
                print(f"[{processed:4d}/{len(valid_clips)}] Worker-{worker_id}: {category}/{video_id}/{clip_id}.mp4", end=" ... ")
                
                if status == 'success':
                    fps_info = f"fps={res.get('fps', 0):.1f}" if 'fps' in res else ''
                    codec_info = res.get('codec_info', '')
                    message = res.get('message', '')
                    
                    # 顯示保存了哪些版本
                    saved_versions = []
                    if res.get('high_res_saved', False):
                        saved_versions.append("高分辨率")
                    if res.get('low_res_saved', False):
                        saved_versions.append("低分辨率")
                    
                    version_info = f"[{'+'.join(saved_versions) if saved_versions else '已存在'}]"
                    
                    info_str = f"({codec_info}, {fps_info}, {version_info})"
                    if message:
                        info_str += f" {message}"
                    print(f"✅ 成功 {info_str}")
                    success_count += 1
                elif status == 'skip':
                    print(f"⏭️  跳過 ({res['message']})")
                    skip_count += 1
                elif status == 'fail':
                    codec_info = res.get('codec_info', '')
                    error = res.get('error', 'Unknown error')
                    print(f"❌ 失敗 ({codec_info})")
                    print(f"    錯誤: {error}")
                    fail_count += 1
                    failed_clips.append({
                        'video_id': video_id,
                        'category': category,
                        'clip_id': clip_id,
                        'error': error
                    })
                
            except Exception as e:
                processed += 1
                print(f"[{processed:4d}/{len(valid_clips)}] ❌ 處理錯誤: {e}")
                fail_count += 1
    
    elapsed_time = time.time() - start_time
    
    # Step 6: 生成處理報告
    print()
    print("📊 Step 6: 生成處理報告...")
    
    # 統計各類別的成功片段（檢查兩個目錄）
    category_success_high_res = defaultdict(int)
    category_success_low_res = defaultdict(int)
    total_file_size_high_res = 0
    total_file_size_low_res = 0
    
    for clip in valid_clips:
        # 檢查高分辨率版本
        high_res_file = clip['high_res_output_file']
        if os.path.exists(high_res_file):
            category_success_high_res[clip['category']] += 1
            try:
                total_file_size_high_res += os.path.getsize(high_res_file)
            except:
                pass
        else:
            avi_file = high_res_file.replace('.mp4', '.avi')
            if os.path.exists(avi_file):
                category_success_high_res[clip['category']] += 1
                try:
                    total_file_size_high_res += os.path.getsize(avi_file)
                except:
                    pass
        
        # 檢查低分辨率版本
        low_res_file = clip['low_res_output_file']
        if os.path.exists(low_res_file):
            category_success_low_res[clip['category']] += 1
            try:
                total_file_size_low_res += os.path.getsize(low_res_file)
            except:
                pass
        else:
            avi_file = low_res_file.replace('.mp4', '.avi')
            if os.path.exists(avi_file):
                category_success_low_res[clip['category']] += 1
                try:
                    total_file_size_low_res += os.path.getsize(avi_file)
                except:
                    pass
    
    # 生成詳細報告
    report_file = os.path.join(high_res_output_dir, "sampling_report_dual_resolution_25frames.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("採樣數據集處理報告 - 3840x1920分辨率，25幀，雙版本\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"高分辨率版本: 3840x1920 (原始分辨率)\n")
        f.write(f"低分辨率版本: 1024x512 (縮放後)\n")
        f.write(f"採樣幀數: 25幀\n")
        f.write(f"Workers 數量: {NUM_WORKERS}\n")
        f.write(f"處理時間: {elapsed_time:.2f} 秒\n")
        f.write(f"已存在片段（跳過）: {len(existing_clips)}\n")
        f.write(f"跳過非3840x1920片段: {skipped_wrong_resolution}\n")
        f.write(f"跳過不健康片段: {skipped_unhealthy}\n")
        f.write(f"成功採樣: {success_count}\n")
        f.write(f"跳過（處理中已存在）: {skip_count}\n")
        f.write(f"採樣失敗: {fail_count}\n")
        f.write(f"總計處理: {success_count + skip_count + fail_count}\n")
        f.write(f"高分辨率總文件大小: {total_file_size_high_res / (1024**3):.2f} GB\n")
        f.write(f"低分辨率總文件大小: {total_file_size_low_res / (1024**3):.2f} GB\n")
        f.write(f"總文件大小: {(total_file_size_high_res + total_file_size_low_res) / (1024**3):.2f} GB\n\n")
        
        f.write("各類別成功片段數量 (高分辨率):\n")
        f.write("-" * 40 + "\n")
        for category, count in sorted(category_success_high_res.items()):
            f.write(f"{category:20s}: {count:4d}\n")
        
        f.write("\n各類別成功片段數量 (低分辨率):\n")
        f.write("-" * 40 + "\n")
        for category, count in sorted(category_success_low_res.items()):
            f.write(f"{category:20s}: {count:4d}\n")
        
        if missing_videos:
            f.write(f"\n缺少的視頻文件 ({len(missing_videos)} 個):\n")
            f.write("-" * 40 + "\n")
            for video in sorted(missing_videos):
                f.write(f"{video}\n")
        
        if wrong_resolution_videos:
            f.write(f"\n分辨率不符的視頻文件 ({len(wrong_resolution_videos)} 個):\n")
            f.write("-" * 40 + "\n")
            for video in sorted(wrong_resolution_videos):
                f.write(f"{video}\n")
        
        if corrupted_videos:
            f.write(f"\n損壞的視頻文件 ({len(corrupted_videos)} 個):\n")
            f.write("-" * 40 + "\n")
            for video in sorted(corrupted_videos):
                f.write(f"{video}\n")
        
        if failed_clips:
            f.write(f"\n採樣失敗的片段 ({len(failed_clips)} 個):\n")
            f.write("-" * 40 + "\n")
            for clip in failed_clips:
                f.write(f"{clip['category']}/{clip['video_id']}/{clip['clip_id']}.mp4\n")
                f.write(f"  錯誤: {clip['error']}\n")
    
    print(f"✅ 已生成處理報告: {report_file}")
    
    # 最終統計
    print()
    print("=" * 80)
    print("🎉 雙分辨率25幀視頻採樣完成！")
    print("=" * 80)
    print(f"🎯 高分辨率版本: 3840x1920 (原始)")
    print(f"🎯 低分辨率版本: 1024x512 (縮放)")
    print(f"📷 採樣幀數: 25幀")
    print(f"👥 Workers: {NUM_WORKERS}")
    print(f"⏱️  總處理時間: {elapsed_time:.2f} 秒")
    print(f"🏃 平均處理速度: {len(valid_clips) / elapsed_time:.1f} 片段/秒")
    print(f"📁 已存在片段（跳過）: {len(existing_clips)}")
    print(f"⚠️  跳過非3840x1920片段: {skipped_wrong_resolution}")
    print(f"🏥 跳過不健康片段: {skipped_unhealthy}")
    print(f"✅ 成功採樣: {success_count} 個片段")
    print(f"⏭️  跳過（處理中已存在）: {skip_count} 個片段")
    print(f"❌ 採樣失敗: {fail_count} 個片段")
    print(f"💾 高分辨率總大小: {total_file_size_high_res / (1024**3):.2f} GB")
    print(f"💾 低分辨率總大小: {total_file_size_low_res / (1024**3):.2f} GB")
    print(f"💾 總文件大小: {(total_file_size_high_res + total_file_size_low_res) / (1024**3):.2f} GB")
    print(f"📁 高分辨率目錄: {high_res_output_dir}")
    print(f"📁 低分辨率目錄: {low_res_output_dir}")
    
    print(f"\n📊 各類別片段數量 (高分辨率):")
    for category, count in sorted(category_success_high_res.items()):
        print(f"   {category:20s}: {count:4d}")
    
    print(f"\n📊 各類別片段數量 (低分辨率):")
    for category, count in sorted(category_success_low_res.items()):
        print(f"   {category:20s}: {count:4d}")
    
    if success_count + fail_count > 0:
        print(f"\n🎯 成功率: {(success_count / (success_count + fail_count) * 100):.1f}%")
    
    # 生成訓練用的文件列表（兩個版本）
    print()
    print("📝 Step 7: 生成訓練用文件列表...")
    
    # 高分辨率文件列表
    high_res_train_list = os.path.join(high_res_output_dir, "train_list_high_res_25frames.txt")
    with open(high_res_train_list, 'w', encoding='utf-8') as f:
        f.write("# 訓練用視頻文件列表 - 高分辨率版本\n")
        f.write("# 格式: category/video_id/clip_id.mp4\n")
        f.write("# 分辨率: 3840x1920, 幀數: 25\n")
        
        for category in sorted(category_success_high_res.keys()):
            category_dir = os.path.join(high_res_output_dir, category)
            if os.path.exists(category_dir):
                for video_id in sorted(os.listdir(category_dir)):
                    video_dir = os.path.join(category_dir, video_id)
                    if os.path.isdir(video_dir):
                        for filename in sorted(os.listdir(video_dir)):
                            if filename.endswith(('.mp4', '.avi')):
                                rel_path = f"{category}/{video_id}/{filename}"
                                f.write(f"{rel_path}\n")
    
    # 低分辨率文件列表
    low_res_train_list = os.path.join(low_res_output_dir, "train_list_low_res_25frames.txt")
    with open(low_res_train_list, 'w', encoding='utf-8') as f:
        f.write("# 訓練用視頻文件列表 - 低分辨率版本\n")
        f.write("# 格式: category/video_id/clip_id.mp4\n")
        f.write("# 分辨率: 1024x512, 幀數: 25\n")
        
        for category in sorted(category_success_low_res.keys()):
            category_dir = os.path.join(low_res_output_dir, category)
            if os.path.exists(category_dir):
                for video_id in sorted(os.listdir(category_dir)):
                    video_dir = os.path.join(category_dir, video_id)
                    if os.path.isdir(video_dir):
                        for filename in sorted(os.listdir(video_dir)):
                            if filename.endswith(('.mp4', '.avi')):
                                rel_path = f"{category}/{video_id}/{filename}"
                                f.write(f"{rel_path}\n")
    
    print(f"✅ 已生成高分辨率訓練列表: {high_res_train_list}")
    print(f"✅ 已生成低分辨率訓練列表: {low_res_train_list}")
    
    return fail_count == 0

if __name__ == "__main__":
    success = prepare_sampled_dataset()
    exit(0 if success else 1)