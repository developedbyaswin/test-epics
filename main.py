from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model and columns
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Apply same encoding as training
        input_df = pd.get_dummies(input_df)

        # Align with training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}