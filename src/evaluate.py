import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from env import FlashcardEnv
from agent import DQNAgent
from simulator import UserSimulator
from baselines import RandomScheduler, LeitnerScheduler, SM2Scheduler, HLRScheduler
from decision_quality_metrics import compute_decision_quality


# Constants for evaluation
INTERVALS = [1, 2, 4, 7, 15, 30, 60]
NUM_CARDS = 50
SIM_DAYS = 180  # Simulate 6 months of learning

def run_simulation(scheduler_type, agent_path=None, seed=42):
    """
    Simulates a 180-day study period for 50 cards using a specific scheduler.
    """
    # Use same seed for UserSimulator and local RNG (Point 1)
    sim = UserSimulator(num_cards=NUM_CARDS, seed=seed)
    rng = np.random.RandomState(seed)
    
    # Initialize scheduler
    if scheduler_type in ["dqn", "dqn_online", "dqn_offline"]:
        agent = DQNAgent(state_dim=3, action_dim=7)
        if agent_path:
            agent.load(agent_path)
        else:
            path = "models/dqn_agent_online.pt" if scheduler_type == "dqn_online" else "models/dqn_agent_offline.pt"
            agent.load(path)
    elif scheduler_type == "random":
        scheduler = RandomScheduler()
    elif scheduler_type == "leitner":
        schedulers = [LeitnerScheduler() for _ in range(NUM_CARDS)]
    elif scheduler_type == "sm2":
        schedulers = [SM2Scheduler() for _ in range(NUM_CARDS)]
    elif scheduler_type == "hlr_anki_original":
        scheduler = HLRScheduler("models/hlr_weights_anki_original.json")
    elif scheduler_type == "hlr_anki_extended":
        scheduler = HLRScheduler("models/hlr_weights_anki_extended.json")
    elif scheduler_type == "hlr_duo_original":
        scheduler = HLRScheduler("models/hlr_weights_duolingo_original.json")
    elif scheduler_type == "hlr_duo_extended":
        scheduler = HLRScheduler("models/hlr_weights_duolingo_extended.json")

    
    # Track when cards are scheduled to be reviewed next (virtual day)
    next_review_day = np.zeros(NUM_CARDS)  # Day 0: all cards ready
    
    total_reviews = 0
    correct_reviews = 0
    
    # Daily logs
    daily_reviews_count = []
    
    # Simulation loop day by day
    for day in range(SIM_DAYS):
        # Find which cards are due today or overdue
        due_cards = np.where(next_review_day <= day)[0]
        
        # Shuffle due cards to simulate realistic study order using local RNG (Point 1)
        rng.shuffle(due_cards)
        
        reviews_today = 0
        for card_id in due_cards:
            # Update current time in simulator to match current day
            sim.current_time = float(day)
            
            # Time elapsed since last review
            t_elapsed = day - sim.last_reviewed[card_id]
            
            # Fetch state for the agent/scheduler
            state = sim.get_state(card_id)
            p_recall = sim.get_recall_probability(card_id, float(day))
            
            # Select action (interval)
            if scheduler_type in ["dqn", "dqn_online", "dqn_offline"]:
                action = agent.select_action(state, evaluate=True)
            elif scheduler_type == "random":
                action = scheduler.select_action(state)
            elif scheduler_type.startswith("hlr"):
                action = scheduler.select_action(state)
            elif scheduler_type == "leitner":
                # Leitner updates based on whether the last review was correct
                # For first review (day=0, consecutive_corrects=0), select Box 0
                history = [h for h in sim.history if h['card_id'] == card_id]
                last_recalled = history[-1]['recalled'] if len(history) > 0 else True
                action = schedulers[card_id].select_action(last_recalled)
            elif scheduler_type == "sm2":
                history = [h for h in sim.history if h['card_id'] == card_id]
                last_recalled = history[-1]['recalled'] if len(history) > 0 else True
                last_p_recall = history[-1]['p_recall'] if len(history) > 0 else 1.0
                action = schedulers[card_id].select_action(last_recalled, last_p_recall)

                
            interval = INTERVALS[action]
            
            # Review card and update simulator
            recalled, new_h = sim.review_card(card_id, t_elapsed)
            
            # Schedule next review
            next_review_day[card_id] = day + interval
            
            total_reviews += 1
            reviews_today += 1
            if recalled:
                correct_reviews += 1
                
        daily_reviews_count.append(reviews_today)
        
    # Final evaluation metrics at the end of SIM_DAYS
    # Calculate average retention rate across all cards on the final day
    final_probabilities = []
    for card_id in range(NUM_CARDS):
        p_recall = sim.get_recall_probability(card_id, float(SIM_DAYS))
        final_probabilities.append(p_recall)
        
    avg_final_retention = np.mean(final_probabilities)
    overall_accuracy = (correct_reviews / total_reviews) if total_reviews > 0 else 0.0
    
    return {
        "scheduler": scheduler_type,
        "total_reviews": total_reviews,
        "overall_accuracy": overall_accuracy,
        "final_retention": avg_final_retention,
        "daily_reviews": daily_reviews_count,
        "history": sim.history
    }

def compare_schedulers():
    print("\n--- Running Comparative Simulation (180 Days, 50 Cards, 15 Seeds) ---")
    seeds = [42, 123, 456, 789, 2024, 7, 13, 99, 888, 1010, 1111, 2025, 2026, 777, 999]
    
    schedulers_to_run = ["random", "leitner", "sm2"]
    if os.path.exists("models/dqn_agent_offline.pt"):
        schedulers_to_run.append("dqn_offline")
    if os.path.exists("models/dqn_agent_online.pt"):
        schedulers_to_run.append("dqn_online")
        
    for hlr_name in ["hlr_anki_original", "hlr_anki_extended", "hlr_duo_original", "hlr_duo_extended"]:
        dataset_key = "anki" if "anki" in hlr_name else "duolingo"
        model_key = "original" if "original" in hlr_name else "extended"
        path_name = f"models/hlr_weights_{dataset_key}_{model_key}.json"
        if os.path.exists(path_name):
            schedulers_to_run.append(hlr_name)
            
    summary_data = []
    dq_summary_data = []
    first_seed_results = {}
    raw_results = {}
    
    # Run simulation for each scheduler over 15 seeds (Point 2)
    for s_type in schedulers_to_run:
        reviews_list = []
        accuracy_list = []
        retention_list = []
        precision_urgent_list = []
        target_deviation_abs_list = []
        mean_signed_deviation_list = []
        pct_reviews_early_list = []
        
        for idx, seed in enumerate(seeds):
            res = run_simulation(s_type, seed=seed)
            reviews_list.append(res["total_reviews"])
            accuracy_list.append(res["overall_accuracy"] * 100.0)
            retention_list.append(res["final_retention"] * 100.0)
            
            # Compute Decision Quality for this seed
            dq = compute_decision_quality(res["history"])
            if dq["precision_urgent"] is not None:
                precision_urgent_list.append(dq["precision_urgent"] * 100.0)
                target_deviation_abs_list.append(dq["target_deviation_abs"])
                mean_signed_deviation_list.append(dq["mean_signed_deviation"])
                pct_reviews_early_list.append(dq["pct_reviews_early"] * 100.0)
            
            if seed == 42:
                first_seed_results[s_type] = res
                
        # Store raw arrays for statistical tests
        raw_results[s_type] = {
            "reviews": reviews_list,
            "retention": retention_list
        }
                
        # Compute mean and standard deviation
        rev_mean = np.mean(reviews_list)
        rev_std = np.std(reviews_list)
        
        acc_mean = np.mean(accuracy_list)
        acc_std = np.std(accuracy_list)
        
        ret_mean = np.mean(retention_list)
        ret_std = np.std(retention_list)
        
        # Compute decision quality stats
        pu_mean = np.mean(precision_urgent_list) if precision_urgent_list else 0.0
        pu_std = np.std(precision_urgent_list) if precision_urgent_list else 0.0
        tda_mean = np.mean(target_deviation_abs_list) if target_deviation_abs_list else 0.0
        tda_std = np.std(target_deviation_abs_list) if target_deviation_abs_list else 0.0
        msd_mean = np.mean(mean_signed_deviation_list) if mean_signed_deviation_list else 0.0
        msd_std = np.std(mean_signed_deviation_list) if mean_signed_deviation_list else 0.0
        pre_mean = np.mean(pct_reviews_early_list) if pct_reviews_early_list else 0.0
        pre_std = np.std(pct_reviews_early_list) if pct_reviews_early_list else 0.0
        
        # Calculate Reviews per 1% retention (Point 4)
        rev_per_ret = rev_mean / ret_mean if ret_mean > 0 else 0.0
        
        summary_data.append({
            "Scheduler": s_type.upper(),
            "Reviews_Mean": rev_mean,
            "Reviews_Std": rev_std,
            "Accuracy_Mean": acc_mean,
            "Accuracy_Std": acc_std,
            "Retention_Mean": ret_mean,
            "Retention_Std": ret_std,
            "Reviews_Per_1Pct_Retention": rev_per_ret
        })
        
        dq_summary_data.append({
            "Scheduler": s_type.upper(),
            "Precision_Urgent_Mean": pu_mean,
            "Precision_Urgent_Std": pu_std,
            "Target_Deviation_Abs_Mean": tda_mean,
            "Target_Deviation_Abs_Std": tda_std,
            "Mean_Signed_Deviation_Mean": msd_mean,
            "Mean_Signed_Deviation_Std": msd_std,
            "Pct_Reviews_Early_Mean": pre_mean,
            "Pct_Reviews_Early_Std": pre_std
        })
        
    summary_df = pd.DataFrame(summary_data)
    
    # Print formatted output table (Point 3)
    print("\n==========================================================================================================")
    print("FINAL EVALUATION TABLE (Mean ± Std over 15 Seeds)")
    print("==========================================================================================================")
    print(f"{'Scheduler':<20} | {'Reviews (mean±std)':<20} | {'Accuracy (mean±std)':<20} | {'Retention (mean±std)':<20} | {'Reviews/1% Retention':<20}")
    print("-" * 110)
    for _, row in summary_df.iterrows():
        rev_str = f"{row['Reviews_Mean']:.1f} ± {row['Reviews_Std']:.1f}"
        acc_str = f"{row['Accuracy_Mean']:.2f}% ± {row['Accuracy_Std']:.2f}%"
        ret_str = f"{row['Retention_Mean']:.2f}% ± {row['Retention_Std']:.2f}%"
        eff_str = f"{row['Reviews_Per_1Pct_Retention']:.2f}"
        print(f"{row['Scheduler']:<20} | {rev_str:<20} | {acc_str:<20} | {ret_str:<20} | {eff_str:<20}")
    print("==========================================================================================================\n")
    
    # Print formatted decision quality table
    print("===========================================================================================================================================")
    print("DECISION QUALITY EVALUATION TABLE (Mean \u00b1 Std over 15 Seeds)")
    print("===========================================================================================================================================")
    print(f"{'Scheduler':<20} | {'Precision (Urgent)':<25} | {'Target Deviation (Abs)':<25} | {'Mean Signed Deviation':<25} | {'% Reviews Early':<25}")
    print("-" * 139)
    for row in dq_summary_data:
        pu_str = f"{row['Precision_Urgent_Mean']:.2f}% \u00b1 {row['Precision_Urgent_Std']:.2f}%"
        tda_str = f"{row['Target_Deviation_Abs_Mean']:.4f} \u00b1 {row['Target_Deviation_Abs_Std']:.4f}"
        msd_str = f"{row['Mean_Signed_Deviation_Mean']:.4f} \u00b1 {row['Mean_Signed_Deviation_Std']:.4f}"
        pre_str = f"{row['Pct_Reviews_Early_Mean']:.2f}% \u00b1 {row['Pct_Reviews_Early_Std']:.2f}%"
        print(f"{row['Scheduler']:<20} | {pu_str:<25} | {tda_str:<25} | {msd_str:<25} | {pre_str:<25}")
    print("===========================================================================================================================================\n")
    
    # Paired t-tests vs SM2 (Point 1)
    if "sm2" in raw_results:
        import scipy.stats as stats
        print("==========================================================================================================")
        print("PAIRED T-TEST SIGNIFICANCE TESTS (vs SM2, N=15 seeds)")
        print("==========================================================================================================")
        
        sm2_rev = raw_results["sm2"]["reviews"]
        sm2_ret = raw_results["sm2"]["retention"]
        
        for dqn_variant in ["dqn_offline", "dqn_online"]:
            if dqn_variant in raw_results:
                dqn_rev = raw_results[dqn_variant]["reviews"]
                dqn_ret = raw_results[dqn_variant]["retention"]
                
                # Paired t-test
                t_stat_rev, p_val_rev = stats.ttest_rel(dqn_rev, sm2_rev)
                t_stat_ret, p_val_ret = stats.ttest_rel(dqn_ret, sm2_ret)
                
                print(f"[{dqn_variant.upper()} vs SM2]")
                print(f" - Reviews:   t-statistic = {t_stat_rev:+.4f}, p-value = {p_val_rev:.6f}")
                print(f" - Retention: t-statistic = {t_stat_ret:+.4f}, p-value = {p_val_ret:.6f}")
                print()
        print("==========================================================================================================\n")
    
    # Save statistics as text file
    summary_df.to_csv("results/comparison_summary.csv", index=False)
    print("Summary CSV saved to results/comparison_summary.csv")
    
    # Generate decision quality markdown report
    os.makedirs("results", exist_ok=True)
    dq_md = f"""# Decision Quality Report

| Scheduler | Precision (Urgent Intervention) mean\u00b1std | Target Deviation (Abs) mean\u00b1std | Mean Signed Deviation mean\u00b1std | % Reviews Early (ôn khi p_recall > 0.85) mean\u00b1std |
| :--- | :---: | :---: | :---: | :---: |
"""
    for row in dq_summary_data:
        pu_str = f"{row['Precision_Urgent_Mean']:.2f}% \u00b1 {row['Precision_Urgent_Std']:.2f}%"
        tda_str = f"{row['Target_Deviation_Abs_Mean']:.4f} \u00b1 {row['Target_Deviation_Abs_Std']:.4f}"
        msd_str = f"{row['Mean_Signed_Deviation_Mean']:.4f} \u00b1 {row['Mean_Signed_Deviation_Std']:.4f}"
        pre_str = f"{row['Pct_Reviews_Early_Mean']:.2f}% \u00b1 {row['Pct_Reviews_Early_Std']:.2f}%"
        dq_md += f"| **{row['Scheduler']}** | {pu_str} | {tda_str} | {msd_str} | {pre_str} |\n"
        
    with open("results/decision_quality_report.md", "w", encoding="utf-8") as f:
        f.write(dq_md)
    print("Decision Quality report saved to results/decision_quality_report.md")
    
    # 1. Plot comparison of total reviews & final retention (using mean values)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sched_names = summary_df["Scheduler"].tolist()
    reviews_means = summary_df["Reviews_Mean"].tolist()
    retention_means = summary_df["Retention_Mean"].tolist()
    
    sns.barplot(x=sched_names, y=reviews_means, ax=axes[0], hue=sched_names, legend=False, palette="Blues_d")
    axes[0].set_title("Mean Total Reviews Required (Lower = Higher Efficiency)")
    axes[0].set_ylabel("Number of Reviews")
    for i, v in enumerate(reviews_means):
        axes[0].text(i, v + 20, f"{v:.1f}", ha='center', fontweight='bold')
        
    sns.barplot(x=sched_names, y=retention_means, ax=axes[1], hue=sched_names, legend=False, palette="Greens_d")
    axes[1].set_title("Mean Final Retention Rate (%)")
    axes[1].set_ylabel("Retention Rate (%)")
    axes[1].set_ylim(0, 110)
    for i, v in enumerate(retention_means):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    axes[0].tick_params(axis='x', labelrotation=30)
    axes[1].tick_params(axis='x', labelrotation=30)
    plt.tight_layout()
    plot_path = "results/scheduler_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Comparison plot saved to {plot_path}")
    
    # 2. Plot Recall Probability over time (using seed 42 results)
    plt.figure(figsize=(12, 6))
    for s_type, res in first_seed_results.items():
        df_hist = pd.DataFrame(res["history"])
        if len(df_hist) > 0:
            df_hist["day"] = df_hist["time"].astype(int)
            daily_acc = df_hist.groupby("day")["recalled"].mean()
            smoothed = daily_acc.rolling(window=10, min_periods=1).mean() * 100
            plt.plot(smoothed, label=s_type.upper(), linewidth=2)
            
    plt.title("Study Session Success Rate Over Time (Seed 42, 10-Day Rolling Avg)")
    plt.xlabel("Simulation Day")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    timeline_plot_path = "results/retention_timeline.png"
    plt.savefig(timeline_plot_path)
    plt.close()
    print(f"Timeline plot saved to {timeline_plot_path}")
    
    # Print existing RL Training Metrics Report
    print("\n==========================================================================================================")
    print("RL TRAINING METRICS REPORT (from results/rl_training_metrics_report.md)")
    print("==========================================================================================================")
    report_path = "results/rl_training_metrics_report.md"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("Report file not found.")
    print("==========================================================================================================\n")

if __name__ == "__main__":
    compare_schedulers()
