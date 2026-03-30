def generate_report(detections):
    """
    detections: list of labels (helmet/head)
    """

    helmet_count = detections.count("helmet")
    head_count = detections.count("head")

    if head_count > 0:
        return f"⚠️ {head_count} worker(s) without helmet detected."
    elif helmet_count > 0:
        return f"✅ All workers wearing helmets."
    else:
        return "No workers detected."