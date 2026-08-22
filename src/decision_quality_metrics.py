import numpy as np

def compute_decision_quality(history, p_recall_threshold=0.7, target_recall=0.85):
    """
    history: list of dicts with 'p_recall' key (logged each time review_card() is called).
    
    Precision (Urgent Intervention): % of reviews that were necessary (p_recall < threshold).
    Reviewing when p_recall >= threshold is considered wasted effort (reviewing too early).
    
    Target Deviation (Abs): mean absolute error between p_recall at review time and target_recall (0.85).
    
    Mean Signed Deviation: mean signed error (positive = tendency to review early, negative = tendency to review late).
    
    Pct Reviews Early: % of reviews conducted when p_recall > 0.85.
    """
    p_recalls = [h['p_recall'] for h in history]
    n_total = len(p_recalls)
    if n_total == 0:
        return {
            "precision_urgent": None, 
            "target_deviation_abs": None, 
            "mean_signed_deviation": None,
            "pct_reviews_early": None,
            "n_reviews": 0
        }
    
    n_urgent = sum(1 for p in p_recalls if p < p_recall_threshold)
    precision_urgent = n_urgent / n_total
    
    signed_deviation = [p - target_recall for p in p_recalls]
    target_deviation_abs = float(np.mean([abs(d) for d in signed_deviation]))
    mean_signed_deviation = float(np.mean(signed_deviation))
    pct_reviews_early = sum(1 for d in signed_deviation if d > 0) / n_total
    
    return {
        "precision_urgent": precision_urgent,
        "target_deviation_abs": target_deviation_abs,
        "mean_signed_deviation": mean_signed_deviation,
        "pct_reviews_early": pct_reviews_early,
        "n_reviews": n_total
    }
