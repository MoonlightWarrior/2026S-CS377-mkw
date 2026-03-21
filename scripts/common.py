import os
import zipfile
import tarfile

import requests
from tqdm import tqdm


def download_file(url, zip_file):
    print(f"Downloading {zip_file}...")
    
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"Failed to download {zip_file}, request code: {response.status_code}")
        exit(1)
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_file, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=zip_file) as bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))
    
    print(f"{zip_file} downloaded!")

def extract_zip(zip_file_path, extract_to_dir):
    os.makedirs(extract_to_dir, exist_ok=True)
    
    print(f"Extracting {zip_file_path} to {extract_to_dir}")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)

    print("Extraction complete!")

def extract_tar(tar_file_path, extract_to_dir):
    os.makedirs(extract_to_dir, exist_ok=True)

    print(f"Extracting {tar_file_path} to {extract_to_dir}")
    with tarfile.open(tar_file_path, 'r:gz') as tar_file:
        tar_file.extractall(path=extract_to_dir)

    print("Extraction complete!")
