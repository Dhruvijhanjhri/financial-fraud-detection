def calculate_risk(probability: float) -> float:
    """Convert a fraud probability to a risk score (0-100)."""
    try:
        score = float(probability) * 100
    except Exception:
        score = 0.0
    return round(score, 2)
