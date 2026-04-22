import os
import time
from pathlib import Path

import dropbox
from requests.exceptions import ChunkedEncodingError, ConnectionError


def _load_dotenv(dotenv_path: str | os.PathLike[str] = ".env") -> None:
    """Minimal .env loader (KEY=VALUE per line).

    - Ignores blank lines and lines starting with '#'
    - Supports optional single/double quotes around values
    - Does not override existing environment variables
    """

    path = Path(dotenv_path)
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (len(value) >= 2) and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).with_name(".env"))

TOKEN = os.environ.get("DROPBOX_TOKEN")
if not TOKEN:
    raise RuntimeError(
        "Missing Dropbox API token. Set DROPBOX_TOKEN in .env (and optionally DROPBOX_SHARED_LINK) "
        "or use the public-link downloader script instead."
    )

# 共享連結（可用環境變數覆蓋）
SHARED_LINK = 'https://www.dropbox.com/scl/fo/bpxvkojbghdc3ssvy7pkt/AECxW_fpg-Lw4AQobfHPhEE?rlkey=empa0cwpx7tex1ypmujwkm0z9&st=l1c6mdk5&dl=0'

dbx = dropbox.Dropbox(TOKEN, timeout=7200)  # 120 min
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

ALLOWED_EXTS = {".mp4"}

download_count = 0
MAX_DOWNLOADS = 300  # 本次執行的目標下載數量 230 videos/53G

try:
    res = dbx.files_list_folder(path="", shared_link=shared_link)

    def process_entries(entries):
        global download_count
        for entry in entries:
            if download_count >= MAX_DOWNLOADS:
                print(f"\n✋ 已達到本次設定上限 {MAX_DOWNLOADS} 部，停止下載。")
                return True
            
            if isinstance(entry, dropbox.files.FileMetadata):
                # 只下載 mp4（不分大小寫）
                _, ext = os.path.splitext(entry.name)
                if ext.lower() not in ALLOWED_EXTS:
                    continue
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