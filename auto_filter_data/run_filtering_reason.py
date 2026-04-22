import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# 過濾配置
FILTER_CONFIG = {
    "cut_detection": {
        "enable": True,
        "cut_single_thd": 0.3,
        "cut_all_thd": 0.2,
    },
    "optical_flow": {
        "enable": True,
        "indicator_key": "q75",
        "min_value": 3.0,
    },
    "perspective_detection": {
        "enable": True,
        "max_value": 0.25,
    },
    "dual_fisheye_detection": {
        "enable": True,
        "max_clip_ratio": 0.9,
    },
    "image_set_detection": {
        "enable": True,
        "min_pixel_mse": 1.0,
    },
    "static_region_detection": {
        "enable": True,
        "min_pixel_mse": 1.0,
        "indicator_keys": [
            "top_pixel_mse_0.2",
            "bottom_pixel_mse_0.2",
        ],
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_list",
        type=str,
        required=True,
        help="Input video list CSV file (category,video_id,clip_id)"
    )
    parser.add_argument(
        "--filter_results_root",
        type=str,
        default="filter_results",
        help="Root directory containing filter results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for all CSV files"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="",
        help="Prefix for output filenames"
    )
    return parser.parse_args()

def read_cut_detection_result(file_path):
    """讀取 cut detection 結果 (兩列數值)"""
    try:
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            return data[0], data[1] if len(data) > 1 else 0.0
        else:
            return data[0, 0], data[0, 1] if data.shape[1] > 1 else 0.0
    except:
        return 1.0, 1.0  # 讀取失敗時假設不通過

def read_single_value_result(file_path):
    """讀取單一數值結果"""
    try:
        with open(file_path, 'r') as f:
            return float(f.read().strip())
    except:
        return 0.0  # 讀取失敗時使用預設值

def read_optical_flow_result(file_path):
    """讀取 optical flow JSON 結果"""
    try:
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('q75', 0.0)
    except:
        return 0.0

def read_static_region_result(file_path):
    """讀取 static region 結果 (CSV格式)"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 第一行是header，第二行是top，第三行是bottom
        if len(lines) >= 3:
            # 解析每行的數值
            top_values = [float(x) for x in lines[1].strip().split(',')]
            bottom_values = [float(x) for x in lines[2].strip().split(',')]
            
            # 找到0.2對應的索引 (第4個值，索引3)
            region_ratios = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            idx_02 = region_ratios.index(0.2)
            
            return top_values[idx_02], bottom_values[idx_02]
        return 0.0, 0.0
    except:
        return 0.0, 0.0

def main():
    args = parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 讀取視頻清單
    try:
        df = pd.read_csv(args.video_list)
        print(f"CSV 欄位: {list(df.columns)}")
    except:
        # 如果沒有標題，手動設定欄位名稱
        df = pd.read_csv(args.video_list, header=None, names=['category', 'video_id', 'clip_id'])
    
    # 檢查是否有正確的欄位
    if 'category' not in df.columns:
        # 嘗試用索引讀取
        if len(df.columns) >= 3:
            df.columns = ['category', 'video_id', 'clip_id']
        else:
            print("錯誤: CSV 檔案格式不正確")
            print(f"發現的欄位: {list(df.columns)}")
            print("請確保 CSV 檔案有 category,video_id,clip_id 三個欄位")
            return
    
    print(f"總共 {len(df)} 個視頻")
    print(f"前5筆資料:")
    print(df.head())
    
    # 初始化各種失敗類型的收集器 (只保留 failed，不要 missing)
    filter_failures = {
        "cut_detection_failed": [],
        "optical_flow_failed": [],
        "perspective_detection_failed": [],
        "dual_fisheye_detection_failed": [],
        "image_set_detection_failed": [],
        "static_region_detection_failed": [],
    }
    
    # 通過所有過濾的視頻
    passed_videos = []
    
    # 統計資訊
    filter_stats = {
        "cut_detection": 0,
        "optical_flow": 0, 
        "perspective_detection": 0,
        "dual_fisheye_detection": 0,
        "image_set_detection": 0,
        "static_region_detection": 0,
        "total_passed": 0,
        "missing_files": 0  # 統計缺失檔案數量但不輸出列表
    }
    
    filter_results_root = Path(args.filter_results_root)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        category = row['category']
        video_id = row['video_id'] 
        clip_id = str(row['clip_id']).split('.')[0]  # 移除可能的.mp4後綴
        
        # 構建檔案路徑
        rel_path = Path(category) / video_id / clip_id
        
        video_info = {
            'category': category,
            'video_id': video_id, 
            'clip_id': clip_id
        }
        
        overall_passed = True
        has_missing_files = False
        
        # 1. Cut Detection
        if FILTER_CONFIG["cut_detection"]["enable"]:
            cut_file = filter_results_root / "cut_detection" / rel_path.with_suffix(".txt")
            if cut_file.exists():
                cut_single, cut_all = read_cut_detection_result(cut_file)
                video_info_detailed = {
                    **video_info,
                    'cut_single': cut_single,
                    'cut_all': cut_all,
                    'cut_single_threshold': FILTER_CONFIG["cut_detection"]["cut_single_thd"],
                    'cut_all_threshold': FILTER_CONFIG["cut_detection"]["cut_all_thd"]
                }
                
                if (cut_single > FILTER_CONFIG["cut_detection"]["cut_single_thd"] or 
                    cut_all > FILTER_CONFIG["cut_detection"]["cut_all_thd"]):
                    filter_failures["cut_detection_failed"].append(video_info_detailed)
                    overall_passed = False
                else:
                    filter_stats["cut_detection"] += 1
            else:
                # 檔案不存在，標記為不通過但不記錄到失敗列表
                overall_passed = False
                has_missing_files = True
        
        # 2. Optical Flow  
        if FILTER_CONFIG["optical_flow"]["enable"]:
            flow_file = filter_results_root / "info_extraction" / rel_path.with_suffix(".json")
            if flow_file.exists():
                q75_value = read_optical_flow_result(flow_file)
                video_info_detailed = {
                    **video_info,
                    'q75_value': q75_value,
                    'min_threshold': FILTER_CONFIG["optical_flow"]["min_value"]
                }
                
                if q75_value < FILTER_CONFIG["optical_flow"]["min_value"]:
                    filter_failures["optical_flow_failed"].append(video_info_detailed)
                    overall_passed = False
                else:
                    filter_stats["optical_flow"] += 1
            else:
                # 檔案不存在，標記為不通過但不記錄到失敗列表
                overall_passed = False
                has_missing_files = True
        
        # 3. Perspective Detection
        if FILTER_CONFIG["perspective_detection"]["enable"]:
            pers_file = filter_results_root / "pers_detection" / rel_path.with_suffix(".txt")
            if pers_file.exists():
                boundary_smoothness = read_single_value_result(pers_file)
                video_info_detailed = {
                    **video_info,
                    'boundary_smoothness': boundary_smoothness,
                    'max_threshold': FILTER_CONFIG["perspective_detection"]["max_value"]
                }
                
                if boundary_smoothness > FILTER_CONFIG["perspective_detection"]["max_value"]:
                    filter_failures["perspective_detection_failed"].append(video_info_detailed)
                    overall_passed = False
                else:
                    filter_stats["perspective_detection"] += 1
            else:
                # 檔案不存在，標記為不通過但不記錄到失敗列表
                overall_passed = False
                has_missing_files = True
        
        # 4. Dual Fisheye Detection  
        if FILTER_CONFIG["dual_fisheye_detection"]["enable"]:
            fisheye_file = filter_results_root / "dual_fisheye_detection" / rel_path.with_suffix(".txt")
            if fisheye_file.exists():
                dual_circle_ratio = read_single_value_result(fisheye_file)
                video_info_detailed = {
                    **video_info,
                    'dual_circle_ratio': dual_circle_ratio,
                    'max_threshold': FILTER_CONFIG["dual_fisheye_detection"]["max_clip_ratio"]
                }
                
                if dual_circle_ratio > FILTER_CONFIG["dual_fisheye_detection"]["max_clip_ratio"]:
                    filter_failures["dual_fisheye_detection_failed"].append(video_info_detailed)
                    overall_passed = False
                else:
                    filter_stats["dual_fisheye_detection"] += 1
            else:
                # 檔案不存在，標記為不通過但不記錄到失敗列表
                overall_passed = False
                has_missing_files = True
        
        # 5. Image Set Detection
        if FILTER_CONFIG["image_set_detection"]["enable"]:
            imgset_file = filter_results_root / "imgset_detection" / rel_path.with_suffix(".txt")
            if imgset_file.exists():
                pixel_mse = read_single_value_result(imgset_file)
                video_info_detailed = {
                    **video_info,
                    'pixel_mse': pixel_mse,
                    'min_threshold': FILTER_CONFIG["image_set_detection"]["min_pixel_mse"]
                }
                
                if pixel_mse < FILTER_CONFIG["image_set_detection"]["min_pixel_mse"]:
                    filter_failures["image_set_detection_failed"].append(video_info_detailed)
                    overall_passed = False
                else:
                    filter_stats["image_set_detection"] += 1
            else:
                # 檔案不存在，標記為不通過但不記錄到失敗列表
                overall_passed = False
                has_missing_files = True
        
        # 6. Static Region Detection
        if FILTER_CONFIG["static_region_detection"]["enable"]:
            static_file = filter_results_root / "static_region_detection" / rel_path.with_suffix(".txt")
            if static_file.exists():
                top_mse, bottom_mse = read_static_region_result(static_file)
                video_info_detailed = {
                    **video_info,
                    'top_mse': top_mse,
                    'bottom_mse': bottom_mse,
                    'min_threshold': FILTER_CONFIG["static_region_detection"]["min_pixel_mse"]
                }
                
                if (top_mse < FILTER_CONFIG["static_region_detection"]["min_pixel_mse"] or
                    bottom_mse < FILTER_CONFIG["static_region_detection"]["min_pixel_mse"]):
                    filter_failures["static_region_detection_failed"].append(video_info_detailed)
                    overall_passed = False
                else:
                    filter_stats["static_region_detection"] += 1
            else:
                # 檔案不存在，標記為不通過但不記錄到失敗列表
                overall_passed = False
                has_missing_files = True
        
        # 統計缺失檔案的視頻數量
        if has_missing_files:
            filter_stats["missing_files"] += 1
        
        # 記錄通過所有過濾的視頻
        if overall_passed:
            passed_videos.append(video_info)
            filter_stats["total_passed"] += 1
    
    # 生成檔案名前綴
    prefix = args.output_prefix + "_" if args.output_prefix else ""
    
    # 輸出各種失敗情況的CSV檔案 (只輸出有檢測結果但失敗的)
    output_files = []
    
    for failure_type, failure_list in filter_failures.items():
        if failure_list:  # 只有當有失敗案例時才創建檔案
            df_failure = pd.DataFrame(failure_list)
            output_file = output_dir / f"{prefix}{failure_type}.csv"
            df_failure.to_csv(output_file, index=False)
            output_files.append(str(output_file))
            print(f"已保存 {len(failure_list)} 個 {failure_type} 案例到: {output_file}")
    
    # 輸出通過所有過濾的視頻
    if passed_videos:
        df_passed = pd.DataFrame(passed_videos)
        passed_file = output_dir / f"{prefix}passed_all_filters.csv"
        df_passed.to_csv(passed_file, index=False)
        output_files.append(str(passed_file))
        print(f"已保存 {len(passed_videos)} 個通過所有過濾的視頻到: {passed_file}")
    
    # 輸出統計摘要
    summary_file = output_dir / f"{prefix}filter_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("視頻過濾統計摘要\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"總視頻數: {len(df)}\n")
        f.write(f"缺失檔案的視頻數: {filter_stats['missing_files']}\n")
        f.write(f"有完整檢測結果的視頻數: {len(df) - filter_stats['missing_files']}\n\n")
        
        f.write("各過濾器通過統計:\n")
        f.write(f"Cut Detection 通過: {filter_stats['cut_detection']}\n")
        f.write(f"Optical Flow 通過: {filter_stats['optical_flow']}\n")
        f.write(f"Perspective Detection 通過: {filter_stats['perspective_detection']}\n")
        f.write(f"Dual Fisheye Detection 通過: {filter_stats['dual_fisheye_detection']}\n")
        f.write(f"Image Set Detection 通過: {filter_stats['image_set_detection']}\n")
        f.write(f"Static Region Detection 通過: {filter_stats['static_region_detection']}\n")
        f.write(f"通過所有過濾: {filter_stats['total_passed']}\n\n")
        
        f.write("各類檢測失敗統計 (僅包含有檢測結果但不合格的視頻):\n")
        for failure_type, failure_list in filter_failures.items():
            count = len(failure_list)
            percentage = count / len(df) * 100
            f.write(f"{failure_type}: {count} 個視頻 ({percentage:.2f}%)\n")
        
        f.write(f"\n過濾後保留比例: {filter_stats['total_passed']}/{len(df)} = {filter_stats['total_passed']/len(df)*100:.2f}%\n")
        
        f.write(f"\n生成的檔案:\n")
        for file_path in output_files:
            f.write(f"- {file_path}\n")
    
    print(f"\n統計摘要已保存到: {summary_file}")
    print(f"\n總共生成了 {len(output_files)} 個CSV檔案和1個統計摘要檔案")
    
    # 在終端顯示統計
    print(f"\n過濾統計:")
    print(f"總視頻數: {len(df)}")
    print(f"缺失檔案的視頻數: {filter_stats['missing_files']}")
    print(f"Cut Detection 通過: {filter_stats['cut_detection']}")
    print(f"Optical Flow 通過: {filter_stats['optical_flow']}")
    print(f"Perspective Detection 通過: {filter_stats['perspective_detection']}")
    print(f"Dual Fisheye Detection 通過: {filter_stats['dual_fisheye_detection']}")
    print(f"Image Set Detection 通過: {filter_stats['image_set_detection']}")
    print(f"Static Region Detection 通過: {filter_stats['static_region_detection']}")
    print(f"最終通過所有過濾: {filter_stats['total_passed']}")
    print(f"過濾後保留比例: {filter_stats['total_passed']}/{len(df)} = {filter_stats['total_passed']/len(df)*100:.2f}%")
    
    # 顯示各類檢測失敗的數量
    print(f"\n各類檢測失敗統計:")
    for failure_type, failure_list in filter_failures.items():
        count = len(failure_list)
        if count > 0:
            print(f"{failure_type}: {count} 個視頻")

if __name__ == "__main__":
    main()