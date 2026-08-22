import numpy as np

def compute_decision_quality(history, p_recall_threshold=0.7, target_recall=0.85):
    """
    history: list các dict có key 'p_recall' (đã có sẵn trong sim.history, field được log ở 
    mỗi lần review_card() được gọi).
    
    Precision (Urgent Intervention): trong số các lần đã ôn, bao nhiêu % là ôn đúng lúc thẻ 
    đang ở mức "nguy cơ" (p_recall < threshold) -- ôn khi p_recall >= threshold coi là lãng phí 
    (ôn quá sớm khi còn nhớ tốt).
    
    Target Deviation (Abs): độ lệch tuyệt đối trung bình giữa p_recall lúc ôn và mục tiêu lý tưởng (0.85).
    
    Mean Signed Deviation: trung bình sai lệch có dấu (dương = xu hướng ôn sớm, âm = xu hướng ôn trễ).
    
    Pct Reviews Early: tỷ lệ ôn tập quá sớm (khi p_recall > 0.85).
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
