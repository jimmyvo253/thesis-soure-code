import pandas as pd
import hashlib
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
import os

# Official fingerprint for Anki 100-user dataset (7.1M raw rows)
RAW_ROW_COUNT_EXPECTED = 7102219
RAW_SHA256_EXPECTED = "ff1058fb5fab6bad1593c27fc5b6d8aa9012584e3a5ab2db806a774dd8e09121"
PROCESSED_ROW_COUNT_EXPECTED = 4809123

def check_raw_dataset(path="data/anki_10k_subset_raw.parquet"):
    """Check raw file before preprocessing. Called in preprocess_anki_10k.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    
    # Check rows
    df = pd.read_parquet(path, columns=["card_id"])
    n_rows = len(df)
    if n_rows != RAW_ROW_COUNT_EXPECTED:
        print(f"[WARNING] {path} has {n_rows:,} rows, EXPECTED {RAW_ROW_COUNT_EXPECTED:,}.")
    else:
        print(f"[OK] Raw dataset: {n_rows:,} rows (matches official).")
        
    # Check SHA256
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            sha256.update(chunk)
    file_hash = sha256.hexdigest()
    if file_hash != RAW_SHA256_EXPECTED:
        print(f"[WARNING] {path} checksum mismatch.")
    else:
        print(f"[OK] Raw dataset SHA-256 matched.")

def check_processed_dataset(path="processed/anki_processed.csv"):
    """Check processed file before offline training."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    
    if path.endswith(".parquet"):
        # unified_anki.parquet uses item_id
        df = pd.read_parquet(path, columns=["item_id"])
    else:
        # anki_processed.csv uses group_id
        df = pd.read_csv(path, usecols=["group_id"])
        
    n_rows = len(df)
    if n_rows != PROCESSED_ROW_COUNT_EXPECTED:
        raise RuntimeError(
            f"[WARNING] {path} has {n_rows:,} rows, EXPECTED {PROCESSED_ROW_COUNT_EXPECTED:,}."
        )
    print(f"[OK] Processed dataset: {n_rows:,} rows (matches official).")
