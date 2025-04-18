import os
import pandas as pd
from datetime import datetime
from pathlib import Path
root = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(root, 'feedback.csv')
csv_path = Path(csv_path)
#print(csv_path)

def save_feedback(title: str, model_label: str, confidence: float, user_label: str = "") -> None:
    now = datetime.now()
    if user_label == "":
        outcome_current = True
    else:
        outcome_current = False

    row = {
        "title": title,
        "outcome_label": model_label,
        "confidence": confidence,
        "timestamp": now,
        "user_label": user_label,
        "outcome_current": outcome_current
    }

    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8")

