import pandas as pd
import numpy as np
import os
import time
import sys

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    start_time = time.time()
    
    # Path settings
    duolingo_input = "processed/duolingo_processed.parquet"
    anki_input = "processed/anki_processed.parquet"
    
    output_dir = "processed"
    duolingo_csv_out = os.path.join(output_dir, "unified_duolingo.csv")
    duolingo_parquet_out = os.path.join(output_dir, "unified_duolingo.parquet")
    anki_csv_out = os.path.join(output_dir, "unified_anki.csv")
    anki_parquet_out = os.path.join(output_dir, "unified_anki.parquet")
    
    
    print(f"============================================================")
    print(f"Starting Dataset Standardization Pipeline")
    print(f"============================================================")

    # 1. Load Processed Datasets
    print("\n[Step 1/5] Loading processed datasets...")
    df_duo = pd.read_parquet(duolingo_input)
    df_ank = pd.read_parquet(anki_input)
    print(f"Loaded Duolingo: {len(df_duo):,} rows")
    print(f"Loaded Anki: {len(df_ank):,} rows")

    # 2. Standardize Duolingo
    print("\n[Step 2/5] Standardizing Duolingo dataset...")
    duo_unified = pd.DataFrame()
    duo_unified['dataset'] = np.repeat('duolingo', len(df_duo))
    duo_unified['user_id'] = df_duo['user_id'].astype(str)
    duo_unified['item_id'] = df_duo['lexeme_id'].astype(str)
    duo_unified['review_timestamp'] = df_duo['timestamp'].astype(np.int64)
    duo_unified['elapsed_time_days'] = df_duo['review_gap_days'].astype(np.float64)
    duo_unified['review_count'] = df_duo['review_count'].astype(np.int64)
    
    # Duolingo lag is in seconds, convert to days
    duo_unified['previous_elapsed_time_days'] = (df_duo['previous_elapsed_time'] / 86400.0).astype(np.float64)
    
    duo_unified['observed_recall'] = df_duo['p_recall'].astype(np.float64)
    duo_unified['cumulative_reviews'] = df_duo['cumulative_reviews'].astype(np.int64)
    duo_unified['cumulative_correct'] = df_duo['cumulative_correct'].astype(np.int64)
    duo_unified['historical_accuracy'] = df_duo['historical_accuracy'].astype(np.float64)

    # 3. Standardize Anki
    print("\n[Step 3/5] Standardizing Anki dataset...")
    ank_unified = pd.DataFrame()
    ank_unified['dataset'] = np.repeat('anki', len(df_ank))
    ank_unified['user_id'] = df_ank['user_id'].astype(str)
    
    # Globally unique item_id (user_id + "_" + card_id)
    ank_unified['item_id'] = df_ank['user_id'].astype(str) + "_" + df_ank['card_id'].astype(str)
    
    # Compute relative Unix seconds: day_offset * 86400. 
    # Since same-day reviews are deduplicated in preprocessing, this timestamp is unique per card.
    ank_unified['review_timestamp'] = (df_ank['day_offset'] * 86400).astype(np.int64)
    
    ank_unified['elapsed_time_days'] = df_ank['delta_t'].astype(np.float64)
    ank_unified['review_count'] = df_ank['review_count'].astype(np.int64)
    ank_unified['previous_elapsed_time_days'] = df_ank['previous_interval'].astype(np.float64)
    ank_unified['observed_recall'] = df_ank['y'].astype(np.float64)
    ank_unified['cumulative_reviews'] = df_ank['cumulative_reviews'].astype(np.int64)
    ank_unified['cumulative_correct'] = df_ank['cumulative_correct'].astype(np.int64)
    ank_unified['historical_accuracy'] = df_ank['historical_accuracy'].astype(np.float64)

    # 4. Resolve potential duplicate timestamps (Deduplication check)
    print("\nDeduplicating any potential rounded timestamp duplicates...")
    # Duolingo
    duo_len_before = len(duo_unified)
    duo_unified = duo_unified.sort_values(by=['user_id', 'item_id', 'review_timestamp'])
    duo_unified = duo_unified.drop_duplicates(subset=['user_id', 'item_id', 'review_timestamp'], keep='last').reset_index(drop=True)
    duo_removed = duo_len_before - len(duo_unified)
    if duo_removed > 0:
        print(f" - Removed {duo_removed:,} duplicate timestamp rows from Duolingo.")
        
    # Anki
    ank_len_before = len(ank_unified)
    ank_unified = ank_unified.sort_values(by=['user_id', 'item_id', 'review_timestamp'])
    ank_unified = ank_unified.drop_duplicates(subset=['user_id', 'item_id', 'review_timestamp'], keep='last').reset_index(drop=True)
    ank_removed = ank_len_before - len(ank_unified)
    if ank_removed > 0:
        print(f" - Removed {ank_removed:,} duplicate timestamp rows from Anki.")

    # Re-calculate review_count if rows were dropped to preserve sequence continuity
    if duo_removed > 0:
        print("Re-calculating review_count for Duolingo...")
        group_mask_duo = (duo_unified['user_id'] != duo_unified['user_id'].shift(1)) | (duo_unified['item_id'] != duo_unified['item_id'].shift(1))
        group_mask_duo.iloc[0] = True
        idx_duo = np.where(group_mask_duo)[0]
        rep_duo = np.diff(np.append(idx_duo, len(duo_unified)))
        duo_unified['review_count'] = np.arange(len(duo_unified)) - np.repeat(idx_duo, rep_duo) + 1
        
    if ank_removed > 0:
        print("Re-calculating review_count for Anki...")
        group_mask_ank = (ank_unified['user_id'] != ank_unified['user_id'].shift(1)) | (ank_unified['item_id'] != ank_unified['item_id'].shift(1))
        group_mask_ank.iloc[0] = True
        idx_ank = np.where(group_mask_ank)[0]
        rep_ank = np.diff(np.append(idx_ank, len(ank_unified)))
        ank_unified['review_count'] = np.arange(len(ank_unified)) - np.repeat(idx_ank, rep_ank) + 1

    # 4. Run Validations
    print("\n[Step 4/5] Running validation checks...")
    
    def validate_unified(df, name):
        print(f"Validating {name} standardized dataset...")
        # Empty checks
        assert (df['user_id'] != "").all(), f"[{name}] user_id contains empty string!"
        assert (df['item_id'] != "").all(), f"[{name}] item_id contains empty string!"
        assert df['user_id'].notnull().all(), f"[{name}] user_id contains nulls!"
        assert df['item_id'].notnull().all(), f"[{name}] item_id contains nulls!"
        assert (df['dataset'] == name).all(), f"[{name}] dataset indicator error!"
        
        # Bounds checks
        assert (df['elapsed_time_days'] >= 0.0).all(), f"[{name}] negative elapsed_time_days!"
        assert (df['review_count'] >= 1).all(), f"[{name}] review_count < 1!"
        assert (df['cumulative_reviews'] >= 0).all(), f"[{name}] negative cumulative_reviews!"
        assert (df['cumulative_correct'] >= 0).all(), f"[{name}] negative cumulative_correct!"
        assert (df['cumulative_correct'] <= df['cumulative_reviews']).all(), f"[{name}] cumulative_correct > cumulative_reviews!"
        assert (df['observed_recall'] >= 0.0).all() and (df['observed_recall'] <= 1.0).all(), f"[{name}] observed_recall out of bounds!"
        assert (df['historical_accuracy'] >= 0.0).all() and (df['historical_accuracy'] <= 1.0).all(), f"[{name}] historical_accuracy out of bounds!"
        
        # Lag bounds check
        valid_prev = df['previous_elapsed_time_days'].isnull() | (df['previous_elapsed_time_days'] >= 0.0)
        assert valid_prev.all(), f"[{name}] negative previous_elapsed_time_days!"
        
        # Chronological sorting check
        group_mask = (df['user_id'] != df['user_id'].shift(1)) | (df['item_id'] != df['item_id'].shift(1))
        group_mask.iloc[0] = True
        ts_diff = df['review_timestamp'].diff().fillna(0.0)
        assert (np.where(group_mask, 0.0, ts_diff) >= 0.0).all(), f"[{name}] timestamps are not sorted!"
        
        # Review count continuity check
        rc_diff = df['review_count'].diff().fillna(1.0)
        assert (np.where(group_mask, 1.0, rc_diff) == 1.0).all(), f"[{name}] review_count sequence is not continuous!"
        
        # Unique review sequence check
        assert not df.duplicated(subset=['user_id', 'item_id', 'review_timestamp']).any(), f"[{name}] duplicate review sequences exist!"
        print(f" - {name} validations passed successfully.")

    validate_unified(duo_unified, 'duolingo')
    validate_unified(ank_unified, 'anki')

    # 5. Export Datasets
    print("\n[Step 5/5] Exporting standardized datasets...")
    # Duolingo
    duo_pq_start = time.time()
    duo_unified.to_parquet(duolingo_parquet_out, engine='pyarrow')
    print(f"Saved Duolingo Parquet to {duolingo_parquet_out} in {time.time() - duo_pq_start:.2f}s")
    
    duo_csv_start = time.time()
    duo_unified.to_csv(duolingo_csv_out, index=False)
    print(f"Saved Duolingo CSV to {duolingo_csv_out} in {time.time() - duo_csv_start:.2f}s")
    
    # Anki
    ank_pq_start = time.time()
    ank_unified.to_parquet(anki_parquet_out, engine='pyarrow')
    print(f"Saved Anki Parquet to {anki_parquet_out} in {time.time() - ank_pq_start:.2f}s")
    
    ank_csv_start = time.time()
    ank_unified.to_csv(anki_csv_out, index=False)
    print(f"Saved Anki CSV to {anki_csv_out} in {time.time() - ank_csv_start:.2f}s")

    print(f"\n============================================================")
    print(f"Standardization successfully completed in {time.time() - start_time:.2f} seconds.")
    print(f"============================================================")

if __name__ == "__main__":
    main()
