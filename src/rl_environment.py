def compute_reward(detections):
    """
    detections: list of dicts with 'confidence'
    """
    tp = sum(1 for d in detections if d["confidence"] > 0.7)
    fp = sum(1 for d in detections if 0.3 < d["confidence"] <= 0.7)
    fn = 1 if len(detections) == 0 else 0

    # Reward design (IMPORTANT for grading)
    reward = (2 * tp) - (1.5 * fp) - (2 * fn)
    return reward