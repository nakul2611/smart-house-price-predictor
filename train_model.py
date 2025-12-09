import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# 1. Load your dataset
# Make sure this CSV file exists in the same folder as this script
DATA_PATH = "house_prices.csv"   # change if your file name is different

print(f"📂 Loading data from: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)
print("✅ Data loaded.")
print("Columns:", list(data.columns))

# 2. Features (X) and Target (y)
# We will use the same features that app_web.py expects:
# POSTED_BY, UNDER_CONSTRUCTION, RERA, BHK_NO., BHK_OR_RK,
# SQUARE_FT, READY_TO_MOVE, RESALE, LONGITUDE, LATITUDE

feature_cols = [
    "POSTED_BY",
    "UNDER_CONSTRUCTION",
    "RERA",
    "BHK_NO.",
    "BHK_OR_RK",
    "SQUARE_FT",
    "READY_TO_MOVE",
    "RESALE",
    "LONGITUDE",
    "LATITUDE",
]

target_col = "TARGET(PRICE_IN_LACS)"

X = data[feature_cols].copy()
y = data[target_col]

# 3. Encode simple categorical columns
# POSTED_BY and BHK_OR_RK are objects (string), convert to category codes
if X["POSTED_BY"].dtype == "object":
    X["POSTED_BY"] = X["POSTED_BY"].astype("category").cat.codes

if X["BHK_OR_RK"].dtype == "object":
    X["BHK_OR_RK"] = X["BHK_OR_RK"].astype("category").cat.codes

print("✅ Encoded categorical columns.")

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("📊 Train size:", X_train.shape, " Test size:", X_test.shape)

# 5. Train a smaller Random Forest (to keep file size small)
model = RandomForestRegressor(
    n_estimators=50,      # fewer trees → smaller file
    max_depth=12,         # limit depth → smaller file
    random_state=42,
    n_jobs=-1
)

print("🚀 Training model...")
model.fit(X_train, y_train)
print("✅ Training done.")

# (Optional) Evaluate quickly
score = model.score(X_test, y_test)
print(f"📈 R² score on test set: {score:.4f}")

# 6. Save model with compression
MODEL_PATH = "house_price_model.pkl"
print("💾 Saving compressed model...")
joblib.dump(model, MODEL_PATH, compress=3)

# 7. Show final file size
size_bytes = os.path.getsize(MODEL_PATH)
size_mb = size_bytes / (1024 * 1024)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"📦 File size: {size_mb:.2f} MB")


