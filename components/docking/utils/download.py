from io import BytesIO
from typing import List
from zipfile import ZipFile
from urllib.request import urlopen
import os

#该函数的主要功能是从远程 URL 下载一个 ZIP 格式的模型文件，并将其解压到本地指定目录中，同时返回压缩包中所有文件和目录的名称列表。
def download_and_extract(remote_model_url: str, local_model_dir) -> List[str]:
    resp = urlopen(remote_model_url)
    os.makedirs(local_model_dir, exist_ok=True)
    with ZipFile(BytesIO(resp.read())) as zip_file:
        all_files_and_dirs = zip_file.namelist()
        zip_file.extractall(local_model_dir)
    return all_files_and_dirs
