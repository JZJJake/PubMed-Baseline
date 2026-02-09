import os
import requests
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

USER_AGENT = "Mozilla/5.0 (compatible; PubMedDownloader/1.0; +https://github.com/your-repo)"

def get_session():
    """
    创建一个带有重试策略的 requests.Session。
    """
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    
 
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_file_list(url=BASE_URL):
    """
    获取给定 URL 的 .xml.gz 文件列表。
    """
    print(f"正在从 {url} 获取文件列表...")
    session = get_session()
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        files = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href.endswith(".xml.gz"):
                files.append(href)
        return sorted(files)
    except requests.exceptions.RequestException as e:
        print(f"获取文件列表出错: {e}")
        return []
    finally:
        session.close()

def download_file(filename, session=None, url=BASE_URL, dest_dir=DATA_DIR, max_retries=5):
    """
    支持断点续传下载单个文件，带有更健壮的重试机制。
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    local_path = os.path.join(dest_dir, filename)
    remote_url = url + filename
    

    should_close_session = False
    if session is None:
        session = get_session()
        should_close_session = True
    

    success = False
    for attempt in range(max_retries):
        resume_header = {}
        mode = 'wb'
        current_size = 0
        
  
        if os.path.exists(local_path):
            current_size = os.path.getsize(local_path)
            if current_size > 0:
                resume_header = {'Range': f'bytes={current_size}-'}
                mode = 'ab'

        try:
    
            try:
                head_response = session.head(remote_url, timeout=30)
       
                if head_response.status_code >= 400:
                     total_size = 0 
                else:
                    total_size = int(head_response.headers.get('content-length', 0))
            except requests.exceptions.RequestException:
 
                total_size = 0

            if total_size > 0 and current_size >= total_size:
                print(f"跳过 {filename} (已下载完成)。")
                success = True
                break

            response = session.get(remote_url, stream=True, headers=resume_header, timeout=60)
            response.raise_for_status()
            

            if response.status_code == 200:
                if current_size > 0:
                    print(f"服务器不支持续传，重新下载 {filename}...")
                current_size = 0
                mode = 'wb'
            elif response.status_code == 206:
                # 续传成功
                pass
            
            # 更新 total_size，如果之前 HEAD 失败或者不准确
            if total_size == 0:
                content_length = response.headers.get('content-length')
                if content_length:
                    total_size = int(content_length) + current_size
            
            block_size = 32 * 1024 * 1024 

           
            with open(local_path, mode) as f:
                with tqdm(
                    desc=filename,
                    initial=current_size,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False # 下载完成后清除进度条
                ) as bar:
                    for data in response.iter_content(block_size):
                        if not data:
                            break
                        size = f.write(data)
                        bar.update(size)
            
            if total_size > 0 and os.path.getsize(local_path) < total_size:
                raise requests.exceptions.ChunkedEncodingError("下载不完整")
            
            print(f"下载完成: {filename}")
            success = True
            break # 成功则退出重试循环

        except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError, ConnectionError) as e:
            wait_time = 5 * (2 ** attempt)
            print(f"\n下载 {filename} 出错: {type(e).__name__} - {e}")
            print(f"将在 {wait_time} 秒后重试 ({attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
            continue
            
        except Exception as e:
             print(f"\n发生未知错误: {type(e).__name__} - {e}")
             break # 未知错误则停止重试

    if should_close_session:
        session.close()

    if not success:
        print(f"\n下载 {filename} 失败，已达到最大重试次数。")
        return None
    return local_path

def sync_files(limit=None):
    """
    同步所有 .xml.gz 文件。
    """
    files = get_file_list()
    if not files:
        print("未找到可下载的文件。")
        return

    print(f"发现 {len(files)} 个文件。")
    
    session = get_session()
    
    count = 0
    success_count = 0
    fail_count = 0
    
    try:
        for file in files:
            if limit and count >= limit:
                break
                
            result = download_file(file, session=session)
            
            if result:
                 success_count += 1
            else:
                 fail_count += 1
                 print(f"跳过 {file} 由于下载失败。")
            
            count += 1
            
            time.sleep(2)
    finally:
        session.close()
        
    print(f"\n同步完成。成功: {success_count}, 失败: {fail_count}")
