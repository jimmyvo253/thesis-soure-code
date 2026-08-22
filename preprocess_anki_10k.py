import pandas as pd
import numpy as np
import os
import time
import sys
from dataset_fingerprint import check_raw_dataset

check_raw_dataset()

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    start_time = time.time()
    
    # Path settings
    raw_input_path = "data/anki_10k_subset_raw.parquet"
    processed_parquet_out = "processed/anki_processed.parquet"
    processed_csv_out = "processed/anki_processed.csv"
    
    report_artifact_path = r"C:\Users\votan\.gemini\antigravity\brain\21dd1e49-927c-475c-8833-060df0f5c951\anki_10k_preprocessing_report.md"
    report_workspace_path = "anki_10k_preprocessing_report.md"
    
    print(f"============================================================")
    print(f"Starting Preprocessing for Anki 10k Dataset (100 Users)")
    print(f"============================================================")

    # 1. Load raw dataset
    print("\n[Step 1/5] Loading raw Parquet dataset...")
    df = pd.read_parquet(raw_input_path)
    orig_rows = len(df)
    print(f"Loaded {orig_rows:,} rows.")

    # 2. Data Cleaning and Filtering
    print("\n[Step 2/5] Cleaning and filtering dataset...")
    
    # Add a temporary index to track original chronological order of logs
    df['temp_review_id'] = np.arange(orig_rows)
    
    # 2.1 Drop exact duplicates (dynamically check for extra columns)
    dup_cols = ['user_id', 'card_id', 'day_offset', 'rating', 'state', 'duration', 'elapsed_days', 'elapsed_seconds']
    for extra_col in ['review_kind', 'button_chosen']:
        if extra_col in df.columns:
            dup_cols.append(extra_col)
            
    df_cleaned = df.drop_duplicates(subset=dup_cols)
    dup_rows = orig_rows - len(df_cleaned)
    print(f" - Exact duplicates dropped: {dup_rows:,}")
    
    # 2.2 Drop missing values in critical columns
    before_missing = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=['user_id', 'card_id', 'day_offset', 'rating'])
    missing_rows = before_missing - len(df_cleaned)
    print(f" - Rows with critical missing values dropped: {missing_rows:,}")
    
    # 2.3 Filter invalid records and manual reschedule logs
    before_invalid = len(df_cleaned)
    # Filter rating to valid Anki ratings [1, 2, 3, 4]
    df_cleaned = df_cleaned[df_cleaned['rating'].isin([1, 2, 3, 4])]
    # Filter state to standard states [0, 1, 2, 3] (0: New, 1: Learning, 2: Review, 3: Relearning)
    df_cleaned = df_cleaned[df_cleaned['state'].isin([0, 1, 2, 3])]
    # Validate offsets and intervals (descriptive elapsed_seconds check)
    df_cleaned = df_cleaned[
        (df_cleaned['day_offset'] >= 0) & 
        ((df_cleaned['elapsed_seconds'] == -1) | (df_cleaned['elapsed_seconds'] >= 0))
    ]
    invalid_rows = before_invalid - len(df_cleaned)
    print(f" - Invalid records dropped (rating/state/offsets): {invalid_rows:,}")
    
    # 2.4 Same-day duplicate reviews (relearning reviews)
    #
    # DECISION: when a card is reviewed multiple times on the same day (e.g.
    # Again -> relearning steps -> eventual pass, all same day_offset), keep
    # the FIRST review of that day.
    #
    # Rationale: this thesis models the forgetting curve / half-life as a
    # function of elapsed time (delta_t) since the previous review. Keeping
    # the first same-day review preserves the true elapsed-time signal from
    # the prior review; keeping the last would overwrite that gap with the
    # card's post-relearning end-of-day state instead. This matches the
    # convention used by the official FSRS benchmark (ankitects/fsrs-benchmark),
    # which also keeps the first same-day review.
    SAME_DAY_DEDUP_KEEP = 'first'

    before_sameday = len(df_cleaned)
    # Sort chronologically using original order (temp_review_id)
    df_cleaned = df_cleaned.sort_values(by=['user_id', 'card_id', 'temp_review_id'])
    df_cleaned = df_cleaned.drop_duplicates(
        subset=['user_id', 'card_id', 'day_offset'], keep=SAME_DAY_DEDUP_KEEP
    )
    sameday_rows = before_sameday - len(df_cleaned)
    print(f" - Same-day duplicate reviews dropped (keeping '{SAME_DAY_DEDUP_KEEP}'): {sameday_rows:,}")
    
    # Drop the temporary index column
    df_cleaned = df_cleaned.drop(columns=['temp_review_id'])

    # 3. Sort records chronologically
    print("\n[Step 3/5] Sorting records chronologically per card...")
    df_sorted = df_cleaned.sort_values(by=['user_id', 'card_id', 'day_offset']).reset_index(drop=True)
    print(f"Final cleaned and sorted dataset: {len(df_sorted):,} rows")

    # 4. Feature Engineering (Prioritizing Readability for Thesis)
    print("\n[Step 4/5] Creating preprocessed columns...")
    
    # observed_recall (y): 1 if rating >= 2, 0 if rating == 1
    df_sorted['y'] = np.where(df_sorted['rating'] >= 2, 1.0, 0.0)
    
    # delta_t: elapsed time in days (elapsed_seconds / 86400.0). For first reviews (-1), set to 0.0.
    df_sorted['delta_t'] = np.where(df_sorted['elapsed_seconds'] >= 0, df_sorted['elapsed_seconds'] / 86400.0, 0.0)

    # 4.1 review_count (1-indexed cumulative index of reviews per card)
    print(" - Computing review_count...")
    df_sorted['review_count'] = df_sorted.groupby(['user_id', 'card_id']).cumcount() + 1
    df_sorted['group_id'] = df_sorted.groupby(['user_id', 'card_id']).ngroup()

    # 4.2 group_mask (Point 2: Clean, Pandas-native boundary check via groupby)
    group_mask = df_sorted['review_count'] == 1

    # 4.3 cumulative_reviews (count of prior reviews)
    print(" - Computing cumulative_reviews...")
    df_sorted['cumulative_reviews'] = df_sorted['review_count'] - 1

    # 4.4 cumulative_correct (count of prior correct reviews)
    # Computes cumulative correct reviews prior to current review
    # (computed as cumsum up to current review minus current review's outcome y)
    print(" - Computing cumulative_correct...")
    df_sorted['cumulative_correct'] = (
        df_sorted.groupby(['user_id', 'card_id'])['y'].cumsum() - df_sorted['y']
    ).astype(np.int64)

    # 4.5 historical_accuracy (ratio of prior successes to prior reviews, default to 0.0)
    print(" - Computing historical_accuracy...")
    df_sorted['historical_accuracy'] = np.where(
        df_sorted['cumulative_reviews'] > 0, 
        df_sorted['cumulative_correct'] / df_sorted['cumulative_reviews'], 
        0.0
    )

    # 4.6 previous_interval (delta_t of the previous review, NaN for first review)
    print(" - Computing previous_interval...")
    df_sorted['previous_interval'] = df_sorted.groupby(['user_id', 'card_id'])['delta_t'].shift(1)

    # 4.7 previous_outcome (y of the previous review, NaN for first review)
    print(" - Computing previous_outcome...")
    df_sorted['previous_outcome'] = df_sorted.groupby(['user_id', 'card_id'])['y'].shift(1)

    # 5. Validations
    print("\n[Step 5/5] Running validation assertions...")
    
    # 5.1 First reviews must start with zero history seen and correct
    first_reviews = df_sorted[group_mask]
    assert (first_reviews['cumulative_reviews'] == 0).all(), "Assertion failed: first reviews have cumulative_reviews > 0!"
    assert (first_reviews['cumulative_correct'] == 0).all(), "Assertion failed: first reviews have cumulative_correct > 0!"
    print(" - Assertion PASSED: First reviews have 0 history seen/correct.")

    # 5.2 Non-negative times
    assert (df_sorted['delta_t'] >= 0.0).all(), "Assertion failed: negative delta_t found!"
    print(" - Assertion PASSED: delta_t is non-negative.")

    # 5.3 Review count increases by exactly 1
    rc_diff = df_sorted['review_count'].diff().fillna(1.0)
    assert (np.where(group_mask, 1.0, rc_diff) == 1.0).all(), "Assertion failed: review_count is not contiguous!"
    print(" - Assertion PASSED: review_count increases strictly by 1.")

    # 5.4 Cumulative correct <= Cumulative reviews
    assert (df_sorted['cumulative_correct'] <= df_sorted['cumulative_reviews']).all(), "Assertion failed: cumulative_correct > cumulative_reviews!"
    print(" - Assertion PASSED: cumulative_correct <= cumulative_reviews.")

    # 5.5 Boundaries for probabilities / accuracies
    assert (df_sorted['historical_accuracy'] >= 0.0).all() and (df_sorted['historical_accuracy'] <= 1.0).all(), "Assertion failed: historical_accuracy out of bounds [0, 1]!"
    print(" - Assertion PASSED: historical_accuracy is within [0.0, 1.0].")

    # 5.6 Check chronological sorting (assert directly on day_offset!)
    day_diff = df_sorted['day_offset'].diff().fillna(0.0)
    assert (np.where(group_mask, 0.0, day_diff) >= 0.0).all(), "Assertion failed: day_offset is not sorted chronologically!"
    print(" - Assertion PASSED: day_offset is sorted chronologically per card.")

    # 5.7 Unique review sequence check (assert directly on day_offset!)
    assert not df_sorted.duplicated(subset=['user_id', 'card_id', 'day_offset']).any(), "Assertion failed: duplicate review day exists per card!"
    print(" - Assertion PASSED: No duplicate review logs on the same day for any card.")

    # 5.8 Group size validation (Point 8: verify review_count.max() == group_size)
    group_sizes = df_sorted.groupby(['user_id', 'card_id']).size()
    max_review_counts = df_sorted.groupby(['user_id', 'card_id'])['review_count'].max()
    assert (group_sizes == max_review_counts).all(), "Assertion failed: maximum review_count does not equal group size!"
    print(" - Assertion PASSED: review_count.max() equals group_size.")

    # 5. Export Preprocessed Dataset
    print("\nExporting preprocessed files...")
    os.makedirs("processed", exist_ok=True)
    
    pq_start = time.time()
    df_sorted.to_parquet(processed_parquet_out, engine='pyarrow')
    print(f" - Saved Parquet to {processed_parquet_out} in {time.time() - pq_start:.2f}s")
    
    csv_start = time.time()
    df_sorted.to_csv(processed_csv_out, index=False)
    print(f" - Saved CSV to {processed_csv_out} in {time.time() - csv_start:.2f}s")

    # Generate Preprocessing Report
    print("\nGenerating Preprocessing Report...")
    
    # Calculate dataset characteristics summary statistics (Point 10 & 3 & 4)
    n_users = df_sorted['user_id'].nunique()
    # Unique card histories (user_id, card_id pairs)
    n_card_histories = len(group_sizes)
    n_reviews = len(df_sorted)
    
    median_revs = group_sizes.median()
    mean_revs = group_sizes.mean()
    
    # Median interval (Point 4: calculated on actual reviews where review_count > 1)
    median_interval = df_sorted[df_sorted['review_count'] > 1]['delta_t'].median()
    
    # Recall rate
    recall_rate = df_sorted['y'].mean()
    
    stats_summary_tbl = f"""| Statistic | Value |
|---|---|
| **Users** | {n_users:,} |
| **Unique user-card pairs** | {n_card_histories:,} |
| **Reviews** | {n_reviews:,} |
| **Median reviews/card** | {median_revs:.1f} |
| **Mean reviews/card** | {mean_revs:.1f} |
| **Median interval (actual reviews)** | {median_interval:.2f} days |
| **Recall rate** | {recall_rate:.2%} |"""

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

    numerical_cols = ['day_offset', 'rating', 'state', 'duration', 'elapsed_days', 'elapsed_seconds', 'y', 'delta_t', 'review_count', 'cumulative_reviews', 'cumulative_correct', 'historical_accuracy', 'previous_interval', 'previous_outcome', 'group_id']
    stats_tbl = df_to_markdown(df_sorted[numerical_cols].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    null_tbl = df_to_markdown(df_sorted.isnull().sum().to_frame(name="Missing/NaN Count"))

    report_content = f"""# Anki 10k Multi-User Subset Preprocessing Report (Updated)

This report documents the preprocessing results for the 100-user subset extracted from the `anki-revlogs-10k` dataset.

---

## 1. Dataset Dimensions and Summary

| Metric | Value |
|---|---|
| **Original Review Logs** | {orig_rows:,} |
| **Exact Duplicates Dropped** | {dup_rows:,} |
| **Missing Values Dropped** | {missing_rows:,} |
| **Invalid Records Filtered** | {invalid_rows:,} |
| **Same-Day Relearning Reviews Dropped** | {sameday_rows:,} |
| **Final Preprocessed Review Logs** | {len(df_sorted):,} |

### Notes on Data Cleaning Decisions:
1. **Invalid Records Filtered:** Excluded ratings outside of `[1, 2, 3, 4]` and states other than `[0, 1, 2, 3]` (such as state 4: preview logs, which do not trigger scheduling changes).
2. **Same-Day Relearning Reviews:** If a card is reviewed multiple times on the same day (e.g. due to relearning steps after a failure), only the **first** review of that day is kept. This preserves the true elapsed-time signal (`delta_t`) from the previous review, which this thesis relies on for forgetting-curve/half-life modeling, and matches the convention used by the official FSRS benchmark (`ankitects/fsrs-benchmark`).
3. **First Review Interval (`delta_t`):** For the first review of each card, where no prior history exists (`elapsed_seconds = -1`), the interval `delta_t` is set to `0.0`. This represents the initial study day (epoch 0) and is a deliberate modeling convention for this dataset.

---

## 2. Spaced Repetition Dataset Characteristics

This table summarizes key characteristics of the preprocessed dataset, matching standard publication formats for spaced repetition research:

{stats_summary_tbl}

---

## 3. Null Value Distribution

{null_tbl}

> [!NOTE]
> The only missing values (`NaN`) in the processed dataset are in the lag features: `previous_interval` and `previous_outcome`. These naturally occur on the first review of each card, where no prior history exists.

---

## 4. Preprocessed Feature Statistics

{stats_tbl}

---

## 5. Verification Assertions Checked
The following checks were verified successfully on the output dataset:
- [x] **Complete History Starting Points**: First reviews of *every* card for *every* user have `cumulative_reviews == 0` and `cumulative_correct == 0`.
- [x] **Non-negative Gaps**: All `delta_t` gaps are non-negative.
- [x] **Review Count Contiguity**: `review_count` increases sequentially by 1 for each card history.
- [x] **Sequence Correctness**: `cumulative_correct <= cumulative_reviews` holds true for all rows.
- [x] **Accuracies Bounds**: `historical_accuracy` lies within `[0.0, 1.0]`.
- [x] **Chronological Sorting**: Review history is sorted in ascending order of `day_offset` per card (verified directly on `day_offset`).
- [x] **Uniqueness**: No duplicate review records exist on the same day for a single user-card combination (verified directly on `day_offset`).
- [x] **Max Review Index Validation**: The maximum `review_count` of each card equals its group size.
"""

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
    print(f"Preprocessing completed successfully in {time.time() - start_time:.2f} seconds.")
    print(f"============================================================")

if __name__ == "__main__":
    main()
