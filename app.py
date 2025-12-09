import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

print("📌 app.py is running...\n")

# 1️⃣ LOAD DATA
data = pd.read_csv("house_prices.csv")
print("✅ Data Loaded Successfully!\n")

print("🔹 First 5 rows of data:")
print(data.head(), "\n")

print("🔹 Columns in dataset:")
print(list(data.columns), "\n")

# 2️⃣ SELECT FEATURES & TARGET
target_col = "TARGET(PRICE_IN_LACS)"

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

# keep only needed columns and drop rows with missing values
data_model = data[feature_cols + [target_col]].dropna()

X = data_model[feature_cols]
y = data_model[target_col]

print("✅ Data prepared for modelling. Rows:", len(X), "\n")

# 3️⃣ DEFINE NUMERIC & CATEGORICAL COLUMNS
numeric_features = [
    "UNDER_CONSTRUCTION",
    "RERA",
    "BHK_NO.",
    "SQUARE_FT",
    "READY_TO_MOVE",
    "RESALE",
    "LONGITUDE",
    "LATITUDE",
]

categorical_features = [
    "POSTED_BY",
    "BHK_OR_RK",
]

# 4️⃣ PREPROCESSOR
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 5️⃣ MODEL
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

# 6️⃣ PIPELINE (preprocessing + model)
regressor = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model),
])

# 7️⃣ TRAIN–TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Training the model...")
regressor.fit(X_train, y_train)

# 8️⃣ EVALUATE
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(((y_test - y_pred) ** 2).mean())

print("\n📊 Evaluation:")
print(f"   MAE  : {mae:.2f} (Lacs)")
print(f"   RMSE : {rmse:.2f} (Lacs)")
print(f"   R²   : {r2:.3f}\n")

# 9️⃣ PREDICT FOR A NEW HOUSE
new_house = pd.DataFrame([{
    "POSTED_BY": "Owner",       # "Owner" / "Dealer" / "Builder"
    "UNDER_CONSTRUCTION": 0,    # 0 or 1
    "RERA": 1,                  # 0 or 1
    "BHK_NO.": 3,               # e.g. 1,2,3,4
    "BHK_OR_RK": "BHK",         # "BHK" or "RK"
    "SQUARE_FT": 1200,          # area in sq ft
    "READY_TO_MOVE": 1,         # 0 or 1
    "RESALE": 1,                # 0 or 1
    "LONGITUDE": 77.59,         # put any valid longitude from your data
    "LATITUDE": 12.97,          # put any valid latitude from your data
}])

pred_price = regressor.predict(new_house)[0]
print("🏠 Predicted price for example house:")
print(f"   {pred_price:.2f} Lacs")



