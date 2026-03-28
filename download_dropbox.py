import dropbox
import os
import time
from dropbox.exceptions import ApiError
from requests.exceptions import ConnectionError, ChunkedEncodingError

# 1. 貼入你的 Token
TOKEN = 'sl.u.AGWqAt1EnS8wf_kjEHKO915lFSTylE0KrrLQh_6_A0kU4NUG0qcPTYxU1diOGw8pO0RbiILj_UgW2Bnw7BMPVxemlnc-XYyruPGvadZ9EoyVnURHOCTy_X-uSW6jVyCf1Brjo2xiDlARUAeA8AphNvW-fHdAwGNVtibC-iWiL1kkqfzutEQAE0IckkH9ZoH1nQXEEd-tnDXDZFZT_GF53vzGrpzemO_dPfvj0lrCFhhDi29wQVuFQLlTB2b8sRFqbQXdvsMA3Vp3fc_hLW3p_E33N9UMokWYKM5SCgFcAmO-vPRHcVgxnDXG8NwgItZHeT1WypoqxzVkzs9Q-0_CJC3oPmP5l4vdPM9RYYiTj-2uT7-06UHsSTXY4yelTSEr_s9lbVr5faameNRUO_15TiuWj2iDnpgXL9l-SaR5LJUyCsmKnfwGygg6hMEBkFbxYf_GqPjjEEOm7pdWNWDaPG7-gXrwrejyn9W9LV3lL3UYZdvWkgJnPAR9xKPimi-ekymCyFRWohcyqTkuhj00dWvUU9ro3bFCGHD2YRocQawHjF32j4K47j4zqZUnhcOsgdaW7DLYHbHl-caw7O-R6-Jc1A_Ny37wjt8625MShaGuT4EkQpSNTQ_qhucKPCYi7BSzXcgS87nIUI1Mcb5Ikxu_kcyAmxhzn8Bfs3zlpHJ4ZT3GZzUarwyT8ZJYi52Nm0LAmGQ2uB4qjEGlIe2XjRd_G45ggGXdETzbCzt4UEf3nYmKPwVxajywjRYAry7_qSHYxuMUat1qLUMVl42VopC16qz68YzpPkHxzhmymGtLqpsmrTQX2EeBWscRgcj5Zb0jKYwkJ7O_iqdn2ioKPe6-ApovEIaTO_xDFHnx8kgkT3IUn2Ev5-_BVYc9vPyMQBbZJnLc7XPo1dcasdlcu-di-JpuFBgHxOS54ZyGMzuKj7RJZfhSfASqF_-8BlqLexMNeOpm91Ev4T1hOhPKhdwpqM4lP3eQ7NOul1VpId1BG39faGHFgLixv8rHbFFtyNis8NPenFGTfMqkFPH_Rg0VmNHlB0noDRThaoGkZ2f0M0k9Vp5KBN_9uEarVbASF8xf72UZO21U8_7-2oYe_f5rC5ECrv9QL69kcXobrJQLpcnWkzNnGAyv9enjJPm1sBrKlK-L-M-qBY1q4WUvKBAcxGQlvOVmsNfSnpPzC6UoKjatoeRwrvDECwjGwnKZwlNeHdd5PYdjIIodp0_heG5gAV5wKwDRi3AFXgzPf3R6Np0eXzqBM9q15adDkSzE0tt5pLCizrLYgX99l5XxBDB9Jv92B1KhV6BtwX0YO-bzKw'
# 2. 你的共享連結
SHARED_LINK = 'https://www.dropbox.com/scl/fo/bpxvkojbghdc3ssvy7pkt/AECxW_fpg-Lw4AQobfHPhEE?rlkey=empa0cwpx7tex1ypmujwkm0z9&st=xq8in5n3&dl=0'

dbx = dropbox.Dropbox(TOKEN, timeout=7200) # 120 min
target_dir = './360-videos'
log_file = 'downloaded_log.txt' # 紀錄檔案名稱

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# --- 紀錄檔處理邏輯 ---

def get_downloaded_history():
    """讀取紀錄檔以及掃描目前資料夾內的檔案"""
    history = set()
    
    # 1. 讀取文字檔紀錄
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                history.add(line.strip())
    
    # 2. 掃描實體資料夾（確保手動丟進去的檔案也被紀錄）
    if os.path.exists(target_dir):
        for filename in os.listdir(target_dir):
            history.add(filename)
            
    return history

def save_to_log(filename):
    """將新下載成功的檔名寫入紀錄"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")

# 初始化已下載清單
downloaded_set = get_downloaded_history()
print(f"📊 目前已下載/紀錄的檔案數: {len(downloaded_set)}")

# --- 下載邏輯 ---

def download_with_retry(dest_path, shared_link_url, file_path, max_retries=5):
    """具備重試機制的下載函數"""
    for i in range(max_retries):
        try:
            # 檢查檔案是否已存在於磁碟上且大小正常
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                return True
            
            dbx.sharing_get_shared_link_file_to_file(dest_path, shared_link_url, path=file_path)
            return True
        except (ConnectionError, ChunkedEncodingError, Exception) as e:
            print(f"  ⚠️ 連線中斷 ({e})，正在進行第 {i+1}/{max_retries} 次重試...")
            time.sleep(5) 
    return False

print("正在獲取 Dropbox 檔案列表...")
shared_link = dropbox.files.SharedLink(url=SHARED_LINK)

download_count = 0
MAX_DOWNLOADS = 500  # 本次執行的目標下載數量 230 videos/53G

try:
    res = dbx.files_list_folder(path="", shared_link=shared_link)

    def process_entries(entries):
        global download_count
        for entry in entries:
            if download_count >= MAX_DOWNLOADS:
                print(f"\n✋ 已達到本次設定上限 {MAX_DOWNLOADS} 部，停止下載。")
                return True
            
            if isinstance(entry, dropbox.files.FileMetadata):
                # 檢查是否已在紀錄中
                if entry.name in downloaded_set:
                    # print(f"  - 跳過已存在檔案: {entry.name}")
                    continue

                print(f"🚀 正在處理第 {download_count + 1} 部: {entry.name} ({entry.size / 1024**2:.2f} MB)")
                dest_path = os.path.join(target_dir, entry.name)
                
                success = download_with_retry(dest_path, SHARED_LINK, "/" + entry.name)
                
                if success:
                    save_to_log(entry.name)   # 寫入紀錄檔
                    downloaded_set.add(entry.name) # 更新記憶體清單
                    download_count += 1
                else:
                    print(f"❌ 檔案 {entry.name} 下載失敗。")
        return False

    stop_signal = process_entries(res.entries)
    
    while res.has_more and not stop_signal:
        res = dbx.files_list_folder_continue(res.cursor)
        stop_signal = process_entries(res.entries)
    
    print(f"\n✅ 任務結束！本次新增下載: {download_count} 部。")

except Exception as e:
    print(f"發生嚴重錯誤: {e}")