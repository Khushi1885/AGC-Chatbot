"""
lead_scoring.py
Loads the trained RandomForestClassifier (lead_scoring_model.pkl) and predicts
lead quality (Hot / Warm / Cold) for a student based on their profile signals.

This is the runtime inference side of the ML component - the model itself is
trained offline once via scripts/train_lead_scoring_model.py.
"""

import os
import joblib
import pandas as pd

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "lead_scoring_model.pkl")
_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(_MODEL_PATH)
    return _bundle


def _infer_discipline(course_name: str) -> str:
    """Maps a free-text course name to one of the training disciplines."""
    name = course_name.lower()
    if any(k in name for k in ["b.tech", "m.tech", "engineering", "b.voc"]):
        return "Engineering"
    if "pharm" in name:
        return "Pharmacy"
    if any(k in name for k in ["mba", "bba", "b.com", "m.com", "management"]):
        return "Management"
    if any(k in name for k in ["bca", "mca", "computer application", "computer science"]):
        return "Computer Applications"
    if any(k in name for k in ["medical lab", "nutrition", "radiology", "operation theatre", "paramedical"]):
        return "Paramedical"
    if any(k in name for k in ["hotel", "hmct", "tourism"]):
        return "Hotel Management"
    if "fashion" in name:
        return "Fashion Design"
    return "Engineering"  # sensible default - most common category


def predict_lead_score(percentage: float, category: str, course_name: str,
                        engagement: int, registered_nest: bool):
    """
    Returns (score: float 0-100, label: 'Hot'/'Warm'/'Cold') for a student.
    Falls back to a safe default if the model can't be loaded, so this never
    crashes the main chat/application flow.
    """
    try:
        bundle = _load_bundle()
        model = bundle["model"]
        cat_encoder = bundle["category_encoder"]
        disc_encoder = bundle["discipline_encoder"]

        discipline = _infer_discipline(course_name)

        # Handle categories/disciplines the encoder hasn't seen before
        category_enc = cat_encoder.transform([category])[0] if category in cat_encoder.classes_ else 0
        discipline_enc = disc_encoder.transform([discipline])[0] if discipline in disc_encoder.classes_ else 0

        features_df = pd.DataFrame(
            [[percentage, category_enc, discipline_enc, min(engagement, 15), int(registered_nest)]],
            columns=bundle["features"],
        )
        label = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence = max(probabilities) * 100

        return round(confidence, 1), label
    except Exception as e:
        print(f"[ML ERROR] Lead scoring failed, using fallback: {e}")
        # Simple safe fallback so the flow never breaks
        if percentage >= 80:
            return 70.0, "Hot"
        elif percentage >= 60:
            return 50.0, "Warm"
        return 30.0, "Cold"
