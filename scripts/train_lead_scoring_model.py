"""
train_lead_scoring_model.py
Generates a realistic synthetic dataset of past "leads" and trains a
RandomForestClassifier to predict lead quality (Hot / Warm / Cold) based on
student signals: 12th percentage, category, course discipline popularity,
chat engagement level, and whether they registered for AGC NEST.

This is a genuine supervised ML component (separate from the LLM agent),
trained with scikit-learn. Run this once to produce lead_scoring_model.pkl,
which the backend loads at runtime via lead_scoring.py.

Run with: python train_lead_scoring_model.py
"""

import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

random.seed(42)
np.random.seed(42)

N_SAMPLES = 800

CATEGORIES = ["General", "OBC", "SC", "ST"]
DISCIPLINES = ["Engineering", "Pharmacy", "Management", "Computer Applications",
               "Paramedical", "Hotel Management", "Fashion Design"]

rows = []
for _ in range(N_SAMPLES):
    percentage = np.clip(np.random.normal(70, 15), 35, 99)
    category = random.choice(CATEGORIES)
    discipline = random.choices(
        DISCIPLINES,
        weights=[30, 15, 20, 15, 10, 5, 5],  # Engineering/Management more popular
    )[0]
    engagement = np.random.poisson(4)  # number of messages exchanged in chat
    engagement = min(engagement, 15)
    registered_nest = random.random() < 0.4  # 40% of leads register for NEST

    # Underlying "true" scoring logic used to generate labels (with noise),
    # simulating how a real admission team might judge lead quality.
    score = 0
    score += (percentage - 35) / (99 - 35) * 40          # up to 40 points for marks
    score += min(engagement, 10) * 3                       # up to 30 points for engagement
    score += 20 if registered_nest else 0                  # 20 points for NEST registration
    score += {"Engineering": 10, "Management": 8, "Computer Applications": 8,
              "Pharmacy": 6, "Paramedical": 5, "Hotel Management": 3, "Fashion Design": 3}[discipline]
    score += np.random.normal(0, 6)  # noise, since real leads aren't perfectly predictable
    score = float(np.clip(score, 0, 100))

    if score >= 70:
        label = "Hot"
    elif score >= 45:
        label = "Warm"
    else:
        label = "Cold"

    rows.append({
        "percentage": round(percentage, 1),
        "category": category,
        "discipline": discipline,
        "engagement": engagement,
        "registered_nest": int(registered_nest),
        "score": round(score, 1),
        "label": label,
    })

df = pd.DataFrame(rows)
print("Sample of generated training data:")
print(df.head(8))
print(f"\nLabel distribution:\n{df['label'].value_counts()}")

# Encode categorical features
cat_encoder = LabelEncoder()
disc_encoder = LabelEncoder()
df["category_enc"] = cat_encoder.fit_transform(df["category"])
df["discipline_enc"] = disc_encoder.fit_transform(df["discipline"])

features = ["percentage", "category_enc", "discipline_enc", "engagement", "registered_nest"]
X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nFeature importances:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")

# Save model + encoders together so the backend can use them at inference time
joblib.dump({
    "model": model,
    "category_encoder": cat_encoder,
    "discipline_encoder": disc_encoder,
    "features": features,
}, "lead_scoring_model.pkl")

print("\nSaved model to lead_scoring_model.pkl")
