import os
import requests
import zipfile

# Thư mục lưu dataset
COCO_DIR = "data/coco"
os.makedirs(COCO_DIR, exist_ok=True)

# Danh sách các tập dữ liệu cần tải
COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

def download_and_extract(url, save_dir):
    filename = url.split("/")[-1]
    filepath = os.path.join(save_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading: {filename}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(filepath, "wb") as file, open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Done: {filename}")
    else:
        print(f"✔ {filename} existed.")

    # Giải nén
    extract_dir = os.path.join(save_dir, filename.split(".")[0])
    if not os.path.exists(extract_dir):
        print(f" Unzipping: {filename}")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(save_dir)
        print(f" Unzipped: {filename}")
    else:
        print(f"✔ {filename} already unzipped.")

# Tải và giải nén tất cả các tệp COCO cần thiết
for name, url in COCO_URLS.items():
    download_and_extract(url, COCO_DIR)

print("Completed")
