import os
import kaggle

# 1. Khai bao thu muc lam viec
target_dir = r"D:\python2\Machine_Learning_dung666_24022301"

# Tao thu muc neu chua ton tai
os.makedirs(target_dir, exist_ok=True)

# 2. Ten dataset tren Kaggle
dataset_name = "mczielinski/bitcoin-historical-data"

print(f"Dang tien hanh tai dataset '{dataset_name}'...")
print(f"Vi tri luu: {target_dir}")

# 3. Goi API de tai va giai nen
kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)

print("Tai va giai nen thanh cong! Ban hay kiem tra thu muc lam viec.")