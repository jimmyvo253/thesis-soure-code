import sys
sys.path.append('src')
from evaluate import run_simulation
import numpy as np
import scipy.stats as stats

seeds_15 = [42, 123, 456, 789, 2024, 7, 13, 99, 888, 1010, 1111, 2025, 2026, 777, 999]

path_800 = "backup_online_800ep/dqn_agent_online.pt"
path_3000 = "models/dqn_agent_online.pt"

print("Running simulations to collect paired samples...")
reviews_800 = []
retention_800 = []
reviews_3000 = []
retention_3000 = []

for idx, seed in enumerate(seeds_15):
    # 800ep
    res_800 = run_simulation("dqn", agent_path=path_800, seed=seed)
    reviews_800.append(res_800["total_reviews"])
    retention_800.append(res_800["final_retention"] * 100.0)
    
    # 3000ep
    res_3000 = run_simulation("dqn", agent_path=path_3000, seed=seed)
    reviews_3000.append(res_3000["total_reviews"])
    retention_3000.append(res_3000["final_retention"] * 100.0)

# Paired t-tests
t_stat_rev, p_val_rev = stats.ttest_rel(reviews_3000, reviews_800)
t_stat_ret, p_val_ret = stats.ttest_rel(retention_3000, retention_800)

print("\n=== PAIRED T-TEST RESULTS (3000ep vs 800ep, N=15) ===")
print(f"Reviews:")
print(f"  - Mean 800ep:  {np.mean(reviews_800):.2f}")
print(f"  - Mean 3000ep: {np.mean(reviews_3000):.2f}")
print(f"  - t-statistic: {t_stat_rev:+.4f}")
print(f"  - p-value:     {p_val_rev:.6f}")

print(f"\nRetention:")
print(f"  - Mean 800ep:  {np.mean(retention_800):.2f}%")
print(f"  - Mean 3000ep: {np.mean(retention_3000):.2f}%")
print(f"  - t-statistic: {t_stat_ret:+.4f}")
print(f"  - p-value:     {p_val_ret:.6f}")
