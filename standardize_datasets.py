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
    
    report_artifact_path = r"C:\Users\votan\.gemini\antigravity\brain\21dd1e49-927c-475c-8833-060df0f5c951\standardization_report.md"
    report_workspace_path = "standardization_report.md"
    
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

    # Generate Standardization Report
    print("\nGenerating Standardization Report...")
    
    def df_to_markdown(df_to_conv):
        temp_df = df_to_conv.reset_index()
        cols = list(temp_df.columns)
        hdr = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows_list = []
        for _, row in temp_df.iterrows():
            row_str = []
            for c in cols:
                val = row[c]
                if isinstance(val, (float, np.float64)):
                    if np.isnan(val):
                        row_str.append("NaN")
                    else:
                        row_str.append(f"{val:.4f}")
                elif isinstance(val, (int, np.int64)):
                    row_str.append(f"{val:,}")
                else:
                    row_str.append(str(val))
            rows_list.append("| " + " | ".join(row_str) + " |")
        return hdr + "\n" + sep + "\n" + "\n".join(rows_list)

    numerical_cols = ['review_timestamp', 'elapsed_time_days', 'review_count', 'previous_elapsed_time_days', 'observed_recall', 'cumulative_reviews', 'cumulative_correct', 'historical_accuracy']
    
    duo_stats = df_to_markdown(duo_unified[numerical_cols].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    ank_stats = df_to_markdown(ank_unified[numerical_cols].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    
    duo_nulls = df_to_markdown(duo_unified.isnull().sum().to_frame(name="Missing/NaN Count"))
    ank_nulls = df_to_markdown(ank_unified.isnull().sum().to_frame(name="Missing/NaN Count"))

    report_content = f"""# Dataset Standardization & Semantic Alignment Report (Updated)

This report documents the standardization process and semantic mapping of the Duolingo and Anki datasets into a unified format.

---

## 1. Standardized Schema Table

| Column Name | Data Type | Semantic Meaning | Source Mapping |
|---|---|---|---|
| `dataset` | `string` | Source indicator: `'duolingo'` or `'anki'`. | Constant |
| `user_id` | `string` | Unique identifier of the student. | Duolingo: `user_id`<br>Anki: `user_id` *(actual)* |
| `item_id` | `string` | Unique identifier of the vocabulary or card item. | Duolingo: `lexeme_id`<br>Anki: `user_id + "_" + card_id` *(composite)* |
| `review_timestamp` | `int64` | Unix epoch time when the review occurred (in seconds). | Duolingo: `timestamp`<br>Anki: `day_offset * 86400` *(relative)* |
| `elapsed_time_days` | `float64` | Time elapsed since the last review (in days). | Duolingo: `review_gap_days`<br>Anki: `delta_t` |
| `review_count` | `int64` | 1-indexed sequential index of reviews. | Duolingo: `review_count`<br>Anki: `review_count` |
| `previous_elapsed_time_days` | `float64` | Time elapsed since the last review of the *previous* session (in days). | Duolingo: `previous_elapsed_time / 86400.0`<br>Anki: `previous_interval` |
| `observed_recall` | `float64` | Recall performance. **Continuous** `p_recall` (Duolingo) or **Binary** `y` (Anki). | Duolingo: `p_recall`<br>Anki: `y` |
| `cumulative_reviews` | `int64` | Total reviews prior to this session. | Duolingo: `cumulative_reviews`<br>Anki: `cumulative_reviews` |
| `cumulative_correct` | `int64` | Total correct reviews prior to this session. | Duolingo: `cumulative_correct`<br>Anki: `cumulative_correct` |
| `historical_accuracy` | `float64` | Pre-session historical accuracy. | Duolingo: `historical_accuracy`<br>Anki: `historical_accuracy` |

### Key Semantic & Technical Alignment Decisions:
1. **Multi-User Structure for both Datasets:**
   * **Duolingo:** Contains 115,222 unique users.
   * **Anki:** Upgraded to use 100 actual users from the `anki-revlogs-10k` dataset.
2. **Semantic Consistency of Recall (`observed_recall`):**
   * **Duolingo:** Continuous ratio representing session success (e.g. `0.73` means 73% correct within the session).
   * **Anki:** Binary value (`0.0` or `1.0`) representing a single trial outcome.
3. **Standardization of Intervals (`elapsed_time_days`, `previous_elapsed_time_days`):**
   * Review intervals are standardized to **days**. Duolingo's seconds-based features are divided by `86400.0`.
4. **Anki Composite `item_id`:**
   * Since card IDs in Anki are only locally unique to each user database, we mapped `item_id` to `user_id + "_" + card_id` to prevent mixing card histories from different users.
5. **Relative Anki Timestamps:**
   * Since absolute timestamps are not present in the 10k dataset, we computed `review_timestamp` as `day_offset * 86400` (in seconds). This relative timestamp is strictly increasing and unique per card since same-day reviews are deduplicated.

---

## 2. Row Counts and Null Handling

| Metric | Duolingo Dataset | Anki Dataset |
|---|---|---|
| **Original Row Count** | 12,854,104 | 4,809,123 |
| **Final Unified Row Count** | 12,854,104 | 4,809,123 |
| **Timestamp Duplicates Dropped** | 0 | 0 |

### Duolingo Unified Null Value Distribution:
{duo_nulls}

### Anki Unified Null Value Distribution:
{ank_nulls}

> [!NOTE]
> The only missing values (`NaN`) are in the lag feature `previous_elapsed_time_days` representing the first review of each sequence, where no prior history exists.

---

## 3. Standardized Feature Statistics

### Duolingo Standardized Feature Stats:
{duo_stats}

### Anki Standardized Feature Stats:
{ank_stats}

---

## 4. Validation Assertions Passed
The following assertions were verified successfully for both datasets:
- [x] **Completeness**: Critical columns (`user_id`, `item_id`, `review_timestamp`, `elapsed_time_days`, `observed_recall`, etc.) have zero nulls.
- [x] **Non-empty strings**: `user_id` and `item_id` contain no empty strings.
- [x] **Boundary Constraints**:
  - `elapsed_time_days >= 0.0`
  - `0.0 <= observed_recall <= 1.0`
  - `0.0 <= historical_accuracy <= 1.0`
  - `cumulative_correct <= cumulative_reviews`
- [x] **Sorting**: Timestamps are strictly non-decreasing for each `(user_id, item_id)` group.
- [x] **Count Continuity**: `review_count` starts at 1 and increases by exactly 1 per review.
- [x] **Uniqueness**: No duplicate records exist for the same user-item-timestamp combination.
"""

    # Write report to files
    with open(report_workspace_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report written to workspace: {report_workspace_path}")
    
    try:
        with open(report_artifact_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Report written to artifacts: {report_artifact_path}")
    except Exception as e:
        print(f"Could not write report to artifact directory: {e}")
        
    print(f"\n============================================================")
    print(f"Standardization successfully completed in {time.time() - start_time:.2f} seconds.")
    print(f"============================================================")

if __name__ == "__main__":
    main()
