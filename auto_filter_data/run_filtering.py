import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

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
        "--output",
        type=str,
        required=True,
        help="Output filtered video list CSV"
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
    
    # 初始化過濾結果
    results = []
    filter_stats = {
        "cut_detection": 0,
        "optical_flow": 0, 
        "perspective_detection": 0,
        "dual_fisheye_detection": 0,
        "image_set_detection": 0,
        "static_region_detection": 0,
        "total_passed": 0
    }
    
    filter_results_root = Path(args.filter_results_root)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        category = row['category']
        video_id = row['video_id'] 
        clip_id = str(row['clip_id']).split('.')[0]  # 移除可能的.mp4後綴
        
        # 構建檔案路徑
        rel_path = Path(category) / video_id / clip_id
        
        passed = True
        fail_reasons = []
        
        # 1. Cut Detection
        if FILTER_CONFIG["cut_detection"]["enable"]:
            cut_file = filter_results_root / "cut_detection" / rel_path.with_suffix(".txt")
            if cut_file.exists():
                cut_single, cut_all = read_cut_detection_result(cut_file)
                if (cut_single > FILTER_CONFIG["cut_detection"]["cut_single_thd"] or 
                    cut_all > FILTER_CONFIG["cut_detection"]["cut_all_thd"]):
                    passed = False
                    fail_reasons.append("cut_detection")
                else:
                    filter_stats["cut_detection"] += 1
            else:
                passed = False
                fail_reasons.append("cut_detection_missing")
        
        # 2. Optical Flow  
        if FILTER_CONFIG["optical_flow"]["enable"] and passed:
            flow_file = filter_results_root / "info_extraction" / rel_path.with_suffix(".json")
            if flow_file.exists():
                q75_value = read_optical_flow_result(flow_file)
                if q75_value < FILTER_CONFIG["optical_flow"]["min_value"]:
                    passed = False
                    fail_reasons.append("optical_flow")
                else:
                    filter_stats["optical_flow"] += 1
            else:
                passed = False  
                fail_reasons.append("optical_flow_missing")
        
        # 3. Perspective Detection
        if FILTER_CONFIG["perspective_detection"]["enable"] and passed:
            pers_file = filter_results_root / "pers_detection" / rel_path.with_suffix(".txt")
            if pers_file.exists():
                boundary_smoothness = read_single_value_result(pers_file)
                if boundary_smoothness > FILTER_CONFIG["perspective_detection"]["max_value"]:
                    passed = False
                    fail_reasons.append("perspective_detection")
                else:
                    filter_stats["perspective_detection"] += 1
            else:
                passed = False
                fail_reasons.append("perspective_detection_missing")
        
        # 4. Dual Fisheye Detection  
        if FILTER_CONFIG["dual_fisheye_detection"]["enable"] and passed:
            fisheye_file = filter_results_root / "dual_fisheye_detection" / rel_path.with_suffix(".txt")
            if fisheye_file.exists():
                dual_circle_ratio = read_single_value_result(fisheye_file)
                if dual_circle_ratio > FILTER_CONFIG["dual_fisheye_detection"]["max_clip_ratio"]:
                    passed = False
                    fail_reasons.append("dual_fisheye_detection")
                else:
                    filter_stats["dual_fisheye_detection"] += 1
            else:
                passed = False
                fail_reasons.append("dual_fisheye_detection_missing")
        
        # 5. Image Set Detection
        if FILTER_CONFIG["image_set_detection"]["enable"] and passed:
            imgset_file = filter_results_root / "imgset_detection" / rel_path.with_suffix(".txt")
            if imgset_file.exists():
                pixel_mse = read_single_value_result(imgset_file)
                if pixel_mse < FILTER_CONFIG["image_set_detection"]["min_pixel_mse"]:
                    passed = False
                    fail_reasons.append("image_set_detection")
                else:
                    filter_stats["image_set_detection"] += 1
            else:
                passed = False
                fail_reasons.append("image_set_detection_missing")
        
        # 6. Static Region Detection
        if FILTER_CONFIG["static_region_detection"]["enable"] and passed:
            static_file = filter_results_root / "static_region_detection" / rel_path.with_suffix(".txt")
            if static_file.exists():
                top_mse, bottom_mse = read_static_region_result(static_file)
                if (top_mse < FILTER_CONFIG["static_region_detection"]["min_pixel_mse"] or
                    bottom_mse < FILTER_CONFIG["static_region_detection"]["min_pixel_mse"]):
                    passed = False
                    fail_reasons.append("static_region_detection")
                else:
                    filter_stats["static_region_detection"] += 1
            else:
                passed = False
                fail_reasons.append("static_region_detection_missing")
        
        # 記錄結果
        if passed:
            results.append({
                'category': category,
                'video_id': video_id, 
                'clip_id': clip_id
            })
            filter_stats["total_passed"] += 1
    
    # 輸出結果
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output, index=False)
    
    print(f"\n過濾統計:")
    print(f"Cut Detection 通過: {filter_stats['cut_detection']}")
    print(f"Optical Flow 通過: {filter_stats['optical_flow']}")
    print(f"Perspective Detection 通過: {filter_stats['perspective_detection']}")
    print(f"Dual Fisheye Detection 通過: {filter_stats['dual_fisheye_detection']}")
    print(f"Image Set Detection 通過: {filter_stats['image_set_detection']}")
    print(f"Static Region Detection 通過: {filter_stats['static_region_detection']}")
    print(f"最終通過所有過濾: {filter_stats['total_passed']}")
    print(f"過濾後保留比例: {filter_stats['total_passed']}/{len(df)} = {filter_stats['total_passed']/len(df)*100:.2f}%")
    
    print(f"\n結果已保存到: {args.output}")

if __name__ == "__main__":
    main()