# ocean_health_model.py
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------- 1) Load entire CSV ----------
# Change this path if needed (e.g., Path("data/data.csv"))
csv_path = Path("data.csv")
if not csv_path.exists() and Path("/mnt/data/data.csv").exists():
    csv_path = Path("/mnt/data/data.csv")  # fallback if you're running in a notebook environment

df = pd.read_csv(csv_path)

print(f"Loaded {len(df):,} rows and {len(df.columns)} columns from: {csv_path}\n")
print("Columns:", list(df.columns), "\n")

# ---------- 2) Correlation ranking ----------
target = "Index_"
numeric_df = df.select_dtypes(include=[np.number]).copy()

if target not in numeric_df.columns:
    raise ValueError(f"Target column '{target}' not found in CSV numeric columns.")

corr = numeric_df.corr(numeric_only=True)[target].drop(labels=[target]).dropna()
corr_rank = corr.reindex(corr.abs().sort_values(ascending=False).index)

print("Top 15 features correlated with Index_ (Pearson):")
print(corr_rank.head(15).to_string(), "\n")

# ---------- 3) Choose features ----------
# Start with the known significant fields; intersect with available columns to be robust.
candidate_features = [
    "BD", "CS", "CW", "FIS", "HAB", "SPP", "AO", "CP", "LE", "trnd_sc"
]
features = [c for c in candidate_features if c in df.columns]

# If you want to automatically use the N most correlated numeric features instead, uncomment:
# N = 12
# features = list(corr_rank.head(N).index)

if not features:
    raise ValueError("No selected features found in the CSV. Check column names.")

print("Using features:", features, "\n")

# ---------- 4) Train/test split & model ----------
model_df = df[features + [target]].dropna()
X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------- 5) Evaluation ----------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("=== Evaluation ===")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}\n")

feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("Feature importances:")
print(feat_imp.to_string(), "\n")

# ---------- 6) (Optional) Save model ----------
out_path = Path("ocean_health_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"model": model, "features": features}, f)
print(f"Model saved to: {out_path.resolve()}")
