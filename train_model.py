import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ==============================
# 1. LOAD DATA
# ==============================
data = pd.read_csv("H:\poshan-web-app-nutritrack-epics\malnutrition_dataset.csv")  # 🔁 change if needed

# ==============================
# 2. CLEAN DATA
# ==============================
data = data.dropna()

# ==============================
# 3. DEFINE TARGET
# ==============================
target_column = "risk_level"

y = data[target_column]
X = data.drop(columns=[target_column])

# ==============================
# 4. HANDLE CATEGORICAL DATA
# ==============================
X = pd.get_dummies(X)

# ==============================
# 5. ENCODE TARGET (if needed)
# ==============================
# If your risk_level is text like: low, medium, high
if y.dtype == "object":
    y = y.map({
        "low": 0,
        "medium": 1,
        "high": 2
    })

# ==============================
# 6. SAVE FEATURE COLUMNS
# ==============================
joblib.dump(X.columns.tolist(), "columns.pkl")

# ==============================
# 7. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 8. MODEL
# ==============================
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss"
)

# ==============================
# 9. TRAIN
# ==============================
model.fit(X_train, y_train)

# ==============================
# 10. EVALUATE
# ==============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.4f}")

# ==============================
# 11. SAVE MODEL
# ==============================
joblib.dump(model, "model.pkl")

print("✅ Model saved as model.pkl")