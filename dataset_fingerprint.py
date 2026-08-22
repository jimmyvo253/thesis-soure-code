import pandas as pd
import hashlib
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
import os

# Số liệu "vân tay" của bộ dữ liệu Anki 100-user CHÍNH THỨC (7.1M dòng raw, đã chốt)
RAW_ROW_COUNT_EXPECTED = 7102219
RAW_SHA256_EXPECTED = "ff1058fb5fab6bad1593c27fc5b6d8aa9012584e3a5ab2db806a774dd8e09121"
PROCESSED_ROW_COUNT_EXPECTED = 4809123

def check_raw_dataset(path="data/anki_10k_subset_raw.parquet"):
    """Kiểm tra file raw trước khi preprocessing. Gọi ở đầu preprocess_anki_10k.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy {path}")
    
    # Check rows
    df = pd.read_parquet(path, columns=["card_id"])
    n_rows = len(df)
    if n_rows != RAW_ROW_COUNT_EXPECTED:
        print(f"[CẢNH BÁO] {path} có {n_rows:,} dòng, KHÔNG khớp bản chính thức ({RAW_ROW_COUNT_EXPECTED:,} dòng).")
        print(f"Mức độ lệch: {abs(n_rows - RAW_ROW_COUNT_EXPECTED):,} dòng.")
    else:
        print(f"[OK] Raw dataset: {n_rows:,} dòng, khớp bản chính thức.")
        
    # Check SHA256
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()
    if file_hash != RAW_SHA256_EXPECTED:
        print(f"[CẢNH BÁO] {path} checksum KHÔNG khớp bản chính thức.")
        print(f"Checksum hiện tại: {file_hash}")
        print(f"Checksum kỳ vọng : {RAW_SHA256_EXPECTED}")
    else:
        print(f"[OK] Raw dataset: Checksum SHA-256 khớp bản chính thức ({file_hash}).")

def check_processed_dataset(path="processed/anki_processed.csv"):
    """Kiểm tra file đã tiền xử lý trước khi train. Gọi ở đầu train_dqn_offline() và 
    train_hlr.py (phần Anki)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy {path}")
    
    if path.endswith(".parquet"):
        # unified_anki.parquet uses item_id
        df = pd.read_parquet(path, columns=["item_id"])
    else:
        # anki_processed.csv uses group_id
        df = pd.read_csv(path, usecols=["group_id"])
        
    n_rows = len(df)
    if n_rows != PROCESSED_ROW_COUNT_EXPECTED:
        raise RuntimeError(
            f"[CẢNH BÁO] {path} có {n_rows:,} dòng, KHÔNG khớp bản chính thức "
            f"({PROCESSED_ROW_COUNT_EXPECTED:,} dòng). Có thể đang huấn luyện trên dữ liệu "
            f"đã bị ghi đè hoặc sai bản!"
        )
    print(f"[OK] Processed dataset: {n_rows:,} dòng, khớp bản chính thức.")

