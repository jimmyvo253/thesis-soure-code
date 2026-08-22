import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Ensure results and models directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Feature Extraction & Dataset Splitting
# -----------------------------------------------------------------------------
def extract_features(df, model_type='original'):
    """
    Extracts features for HLR models:
    - HLR-Original: [1, log2(1 + hc), log2(1 + hw)]
    - HLR-Extended: [1, log2(1 + hc), log2(1 + hw), historical_accuracy, log2(review_count)]
    """
    hc = df['cumulative_correct'].values
    hw = (df['cumulative_reviews'] - df['cumulative_correct']).values
    
    # Base features
    x0 = np.ones(len(df))
    x1 = np.log2(1.0 + hc)
    x2 = np.log2(1.0 + hw)
    
    if model_type == 'original':
        X = np.stack([x0, x1, x2], axis=1)
    else: # extended
        # historical_accuracy = hc / hs (1.0 if hs == 0)
        hs = df['cumulative_reviews'].values
        x3 = np.where(hs > 0, hc / hs, 1.0)
        # log2(review_count)
        x4 = np.log2(df['review_count'].values)
        X = np.stack([x0, x1, x2, x3, x4], axis=1)
        
    y = df['observed_recall'].values
    delta_t = df['elapsed_time_days'].values
    return X, y, delta_t

def split_by_group(df, group_col, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the dataframe based on groups to avoid data leakage.
    """
    unique_groups = df[group_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    
    n_groups = len(unique_groups)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])
    
    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()
    
    return train_df, val_df, test_df

# -----------------------------------------------------------------------------
# 2. PyTorch HLR Model & Loss Function
# -----------------------------------------------------------------------------
class HLRModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Explicit theta parameters (no bias since we include 1 in features)
        self.theta = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))
        
    def forward(self, X, delta_t):
        # h = 2^(theta^T X)
        h = torch.pow(2.0, torch.matmul(X, self.theta))
        # Clip half-life to prevent numerical overflow/underflow
        h = torch.clamp(h, min=0.01, max=36500.0)
        # p = 2^(-delta_t / h)
        p = torch.pow(2.0, -delta_t / h)
        p = torch.clamp(p, min=1e-5, max=1.0 - 1e-5)
        return p, h

def hlr_loss(p, h, y, delta_t, theta, alpha, l2_lambda, is_duo):
    # MSE Loss between predicted p and ground-truth y
    mse = torch.mean((p - y) ** 2)
    
    # Dual objective term: alpha * (h - h_hat)^2 where y is continuous and 0 < y < 1
    dual = torch.tensor(0.0, device=p.device)
    if is_duo and alpha > 0:
        mask = (y > 1e-5) & (y < 1.0 - 1e-5)
        if mask.any():
            y_masked = y[mask]
            delta_t_masked = delta_t[mask]
            h_masked = h[mask]
            
            # h_hat = -delta_t / log2(y)
            h_hat = -delta_t_masked / torch.log2(y_masked)
            h_hat = torch.clamp(h_hat, min=0.01, max=36500.0)
            
            dual = torch.mean((h_masked - h_hat) ** 2)
            
    # L2 regularization
    l2 = l2_lambda * torch.sum(theta ** 2)
    
    return mse + alpha * dual + l2, mse, dual

# -----------------------------------------------------------------------------
# 3. Model Training & Evaluation Logic
# -----------------------------------------------------------------------------
def train_hlr_model(X_train, y_train, dt_train, X_val, y_val, dt_val, 
                    alpha, l2_lambda, lr=0.01, epochs=100, is_duo=False, batch_size=8192):
    
    input_dim = X_train.shape[1]
    model = HLRModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Tensors conversion
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    dt_train_t = torch.FloatTensor(dt_train)
    
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    dt_val_t = torch.FloatTensor(dt_val)
    
    dataset_size = len(X_train)
    
    best_val_loss = float('inf')
    best_theta = None
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(dataset_size)
        
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_dual = 0.0
        batches = 0
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y, batch_dt = X_train_t[indices], y_train_t[indices], dt_train_t[indices]
            
            optimizer.zero_grad()
            p, h = model(batch_x, batch_dt)
            
            loss, mse, dual = hlr_loss(p, h, batch_y, batch_dt, model.theta, alpha, l2_lambda, is_duo)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/Inf detected in loss at epoch {epoch}, batch {i//batch_size}. Skipping update.")
                continue
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse.item()
            epoch_dual += dual.item()
            batches += 1
            
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            p_val, h_val = model(X_val_t, dt_val_t)
            val_loss, val_mse, val_dual = hlr_loss(p_val, h_val, y_val_t, dt_val_t, model.theta, alpha, l2_lambda, is_duo)
            
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_theta = model.theta.clone().detach().cpu().numpy()
            
    # Load best weights
    if best_theta is not None:
        model.theta.data = torch.FloatTensor(best_theta)
        
    return model, best_theta

def calculate_metrics(y_true, y_pred, is_duo=False):
    """
    Calculates specific metrics based on the dataset type.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    brier = np.mean((y_true - y_pred) ** 2)
    
    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'Brier_Score': float(brier)
    }
    
    if is_duo:
        # Coefficient of determination R^2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        metrics['R2'] = float(r2)
    else: # Anki (binary)
        # ROC-AUC (sorting-based implementation to avoid sklearn dependency)
        desc_score_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        n_pos = np.sum(y_true == 1.0)
        n_neg = len(y_true) - n_pos
        
        if n_pos > 0 and n_neg > 0:
            tp = 0
            auc = 0.0
            for label in y_true_sorted:
                if label == 1.0:
                    tp += 1
                else:
                    auc += tp
            auc /= (n_pos * n_neg)
            metrics['ROC_AUC'] = float(auc)
        else:
            metrics['ROC_AUC'] = 0.5
            
    return metrics

# -----------------------------------------------------------------------------
# 4. Diagnostics & Sanity Check Plots
# -----------------------------------------------------------------------------
def run_diagnostics(model, X_test, y_test, dt_test, dataset_name, model_type):
    print(f"\n--- Diagnostics for {dataset_name} ({model_type}) ---")
    theta = model.theta.detach().cpu().numpy()
    print(f"Learned weights (theta): {theta}")
    
    # 1. Monotonicity check
    # theta[1] is for log2(1 + hc) -> must be positive
    # theta[2] is for log2(1 + hw) -> must be negative
    if theta[1] <= 0:
        print(f"[WARNING] Monotonicity failed: theta_1 (correct count weight) is {theta[1]:.4f} (should be > 0).")
    else:
        print(f"[OK] Monotonicity pass: theta_1 = {theta[1]:.4f} > 0.")
        
    if theta[2] >= 0:
        print(f"[WARNING] Monotonicity failed: theta_2 (incorrect count weight) is {theta[2]:.4f} (should be < 0).")
    else:
        print(f"[OK] Monotonicity pass: theta_2 = {theta[2]:.4f} < 0.")
        
    # 2. Predicted Half-life distribution & Stats
    X_test_t = torch.FloatTensor(X_test)
    dt_test_t = torch.FloatTensor(dt_test)
    with torch.no_grad():
        _, h_pred_t = model(X_test_t, dt_test_t)
        h_pred = h_pred_t.cpu().numpy()
        
    h_mean = np.mean(h_pred)
    h_median = np.median(h_pred)
    h_p95 = np.percentile(h_pred, 95)
    h_p99 = np.percentile(h_pred, 99)
    print(f"Half-life stats (days): Mean={h_mean:.2f}, Median={h_median:.2f}, 95th%={h_p95:.2f}, 99th%={h_p99:.2f}")
    
    if h_p99 > 50000.0:
        print(f"[WARNING] Extremely high predicted half-life detected! (99th% = {h_p99:.2f} days)")
        
    # Plot half-life distribution
    plt.figure(figsize=(8, 5))
    plt.hist(h_pred, bins=50, color='#3b82f6', edgecolor='black', alpha=0.7, log=True)
    plt.title(f"Predicted Half-life Distribution - {dataset_name} ({model_type})")
    plt.xlabel("Half-life (Days, Log Scale)")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_dist_path = f"results/hlr_half_life_dist_{dataset_name.lower()}_{model_type}.png"
    plt.savefig(plot_dist_path)
    plt.close()
    print(f"Half-life distribution plot saved to: {plot_dist_path}")
    
    # 3. Forgetting curves plotting for sample states
    # We define 3 sample states:
    # A (Established): hc=10, hw=2, review_count=12, acc=10/12=0.83
    # B (New card): hc=1, hw=0, review_count=1, acc=1.0
    # C (Difficult card): hc=2, hw=5, review_count=7, acc=2/7=0.28
    
    sample_histories = [
        {"name": "Established (hc=10, hw=2)", "hc": 10, "hw": 2, "reviews": 12},
        {"name": "New Card (hc=1, hw=0)", "hc": 1, "hw": 0, "reviews": 1},
        {"name": "Difficult Card (hc=2, hw=5)", "hc": 2, "hw": 5, "reviews": 7}
    ]
    
    t_range = np.linspace(1, 90, 100)
    plt.figure(figsize=(9, 6))
    colors = ['#10b981', '#ef4444', '#f59e0b']
    
    for idx, sh in enumerate(sample_histories):
        # Build feature vector
        if model_type == 'original':
            x = np.array([1.0, np.log2(1.0 + sh["hc"]), np.log2(1.0 + sh["hw"])])
        else:
            acc = sh["hc"] / sh["reviews"]
            x = np.array([1.0, np.log2(1.0 + sh["hc"]), np.log2(1.0 + sh["hw"]), acc, np.log2(sh["reviews"])])
            
        h_val = 2.0 ** np.dot(theta, x)
        h_val = np.clip(h_val, 0.01, 36500.0)
        p_val = 2.0 ** (-t_range / h_val)
        
        # Verify strict monotonicity of forgetting curve
        # (recall probability must decrease as time elapsed increases)
        diffs = np.diff(p_val)
        if (diffs > 1e-6).any():
            print(f"[WARNING] Non-monotonicity in simulated forgetting curve for {sh['name']}!")
            
        plt.plot(t_range, p_val * 100, label=f"{sh['name']} (h={h_val:.2f}d)", color=colors[idx], linewidth=2.5)
        
    plt.title(f"Predicted Forgetting Curves - {dataset_name} ({model_type})")
    plt.xlabel("Elapsed Time (Days)")
    plt.ylabel("Recall Probability (%)")
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_curves_path = f"results/hlr_forgetting_curves_{dataset_name.lower()}_{model_type}.png"
    plt.savefig(plot_curves_path)
    plt.close()
    print(f"Forgetting curves plot saved to: {plot_curves_path}")
    
    return {
        'theta': theta.tolist(),
        'half_life_stats': {
            'mean': float(h_mean),
            'median': float(h_median),
            'p95': float(h_p95),
            'p99': float(h_p99)
        }
    }

# -----------------------------------------------------------------------------
# 5. Main Training Pipeline
# -----------------------------------------------------------------------------
def run_pipeline():
    # Load unified datasets
    anki_path = "processed/unified_anki.parquet"
    duo_path = "processed/unified_duolingo.parquet"
    
    print("============================================================")
    print("Starting Half-Life Regression (HLR) Training Pipeline")
    print("============================================================")
    
    datasets_to_train = []
    if os.path.exists(anki_path):
        datasets_to_train.append(('Anki', anki_path, 'item_id', False))
    else:
        print(f"[Error] Unified Anki dataset not found at {anki_path}")
        
    # if os.path.exists(duo_path):
    #     datasets_to_train.append(('Duolingo', duo_path, 'user_id', True))
    # else:
    #     print(f"[Error] Unified Duolingo dataset not found at {duo_path}")
        
    all_results = {}
    
    for name, path, group_col, is_duo in datasets_to_train:
        print(f"\nProcessing {name} dataset from {path}...")
        
        if name == 'Anki':
            from dataset_fingerprint import check_processed_dataset
            check_processed_dataset(path)
            
        df = pd.read_parquet(path)
        print(f"Loaded {len(df):,} reviews.")
        
        # Capping dataset sizes to make grid search validation efficient
        # Duolingo is huge (12.8M rows), so we sample 15,000 unique users
        # Anki has single user, so we sample 5,000 unique items
        if is_duo and df[group_col].nunique() > 15000:
            print("Capping Duolingo dataset size by sampling 15,000 unique users for speed...")
            sampled_groups = np.random.choice(df[group_col].unique(), 15000, replace=False)
            df = df[df[group_col].isin(sampled_groups)].copy()
            print(f"Capped dataset rows: {len(df):,}")
        elif not is_duo and df[group_col].nunique() > 5000:
            print("Capping Anki dataset size by sampling 5,000 unique cards for speed...")
            sampled_groups = np.random.choice(df[group_col].unique(), 5000, replace=False)
            df = df[df[group_col].isin(sampled_groups)].copy()
            print(f"Capped dataset rows: {len(df):,}")
            
        # Group-based split (70/15/15)
        train_df, val_df, test_df = split_by_group(df, group_col, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        print(f"Splits - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
        
        all_results[name] = {}
        
        # Grid search candidates
        lambdas = [0.001, 0.01, 0.1, 1.0]
        alphas = [0.001, 0.01, 0.1] if is_duo else [0.0] # Dual loss only for Duolingo
        lrs = [0.005, 0.01]
        
        for model_type in ['original', 'extended']:
            print(f"\n--- Training {name} ({model_type}) ---")
            X_train, y_train, dt_train = extract_features(train_df, model_type)
            X_val, y_val, dt_val = extract_features(val_df, model_type)
            X_test, y_test, dt_test = extract_features(test_df, model_type)
            
            # Grid search hyperparameter tuning
            best_val_mse = float('inf')
            best_hyperparams = None
            best_model = None
            
            # Dynamic training configuration
            epochs = 30 if is_duo else 150
            batch_size = 8192 if is_duo else 512
            
            print(f"Running grid search hyperparameter validation (epochs={epochs}, batch_size={batch_size})...")
            for l2 in lambdas:
                for a in alphas:
                    for lr in lrs:
                        model, _ = train_hlr_model(
                            X_train, y_train, dt_train, X_val, y_val, dt_val,
                            alpha=a, l2_lambda=l2, lr=lr, epochs=epochs, is_duo=is_duo, batch_size=batch_size
                        )
                        
                        # Validate
                        model.eval()
                        with torch.no_grad():
                            p_val, _ = model(torch.FloatTensor(X_val), torch.FloatTensor(dt_val))
                            val_mse = np.mean((p_val.cpu().numpy() - y_val) ** 2)

                            
                        if val_mse < best_val_mse:
                            best_val_mse = val_mse
                            best_hyperparams = (l2, a, lr)
                            best_model = model
                            
            l2_opt, a_opt, lr_opt = best_hyperparams
            print(f"Optimal hyperparameters: lambda={l2_opt}, alpha={a_opt}, lr={lr_opt} (Val MSE: {best_val_mse:.6f})")
            
            # Diagnostic plots & verification on test set
            diag_metrics = run_diagnostics(best_model, X_test, y_test, dt_test, name, model_type)
            
            # Calculate final test metrics
            best_model.eval()
            with torch.no_grad():
                p_test, _ = best_model(torch.FloatTensor(X_test), torch.FloatTensor(dt_test))
                y_pred = p_test.cpu().numpy()
                
            test_metrics = calculate_metrics(y_test, y_pred, is_duo)
            print(f"Test performance metrics: {test_metrics}")
            
            all_results[name][model_type] = {
                'theta': diag_metrics['theta'],
                'hyperparams': {
                    'lambda': l2_opt,
                    'alpha': a_opt,
                    'lr': lr_opt
                },
                'metrics': test_metrics,
                'half_life_stats': diag_metrics['half_life_stats']
            }
            
            # Save the optimized weights as a model asset
            weights_filename = f"models/hlr_weights_{name.lower()}_{model_type}.json"
            with open(weights_filename, 'w') as f:
                json.dump({
                    'theta': diag_metrics['theta'],
                    'model_type': model_type,
                    'metrics': test_metrics,
                    'hyperparams': {'lambda': l2_opt, 'alpha': a_opt, 'lr': lr_opt}
                }, f, indent=4)
            print(f"Saved weights to {weights_filename}")
            
    # Save overall summary statistics to results
    summary_path = "results/hlr_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll HLR models trained successfully! Summary saved to {summary_path}")

if __name__ == "__main__":
    run_pipeline()
