from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import numpy as np
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class CustomerData(BaseModel):
    Age: int
    Gender: str
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: int
    model_type: str

# Load dataset for reference categories
try:
    df = pd.read_csv("customer_churn_dataset-training-master.csv")
    logger.info(f"Dataset loaded with shape: {df.shape}")
    logger.info(f"Unique Gender values: {df['Gender'].dropna().unique().tolist()}")
    logger.info(f"Unique Subscription Type values: {df['Subscription Type'].dropna().unique().tolist()}")
    logger.info(f"Unique Contract Length values: {df['Contract Length'].dropna().unique().tolist()}")
except FileNotFoundError as e:
    logger.error(f"Dataset file not found: {e}")
    raise HTTPException(status_code=500, detail="Dataset file not found.")

# Define valid categories from dataset
valid_categories = {
    "Gender": df["Gender"].dropna().unique().tolist(),
    "Subscription Type": df["Subscription Type"].dropna().unique().tolist(),
    "Contract Length": df["Contract Length"].dropna().unique().tolist()
}

# Load models and preprocessing objects
model_files = {
    "logreg": "model/new_model_logistic_regression.pkl",
    "svm": "model/new_model_svm.pkl",
    "knn": "model/new_model_knn.pkl"
}
preprocessing_files = {
    "le_gender": "model/le_gender_new.pkl",
    "le_sub_type": "model/le_sub_type_new.pkl",
    "le_contract": "model/le_contract_new.pkl",
    "scaler": "model/scaler_new.pkl"
}

try:
    models = {key: joblib.load(path) for key, path in model_files.items()}
    preprocessing = {key: joblib.load(path) for key, path in preprocessing_files.items()}
    le_gender = preprocessing["le_gender"]
    le_sub_type = preprocessing["le_sub_type"]
    le_contract = preprocessing["le_contract"]
    scaler = preprocessing["scaler"]
    logger.info("Models and preprocessing objects loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Model or preprocessing file not found: {e}")
    raise HTTPException(status_code=500, detail="Model or preprocessing file not found.")
except Exception as e:
    logger.error(f"Error loading models or preprocessing objects: {e}")
    raise HTTPException(status_code=500, detail=f"Error loading models: {e}")

def preprocess_data(input_data):
    data = input_data.copy()

    logger.info(f"Valid categories at runtime: {valid_categories}")

    # Mapping input field to dataset field names
    field_map = {
        "Gender": "Gender",
        "Subscription_Type": "Subscription Type",
        "Contract_Length": "Contract Length"
    }

    encoders = {
        "Gender": le_gender,
        "Subscription Type": le_sub_type,
        "Contract Length": le_contract
    }

    # Encode categorical fields
    for input_field, dataset_field in field_map.items():
        if dataset_field not in valid_categories:
            raise HTTPException(status_code=500, detail=f"Validation category missing for {dataset_field}")

        input_value = data.get(input_field)
        if input_value not in valid_categories[dataset_field]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for {dataset_field}: {input_value}. Allowed: {valid_categories[dataset_field]}"
            )
        try:
            encoded = encoders[dataset_field].transform([input_value])[0]
            data[input_field] = encoded
            logger.info(f"Encoded {dataset_field}: {encoded} (Classes: {encoders[dataset_field].classes_.tolist()})")
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Encoding failed for {dataset_field}: {e}")

    # Rename input to match original training feature names
    data_renamed = {
        'Age': data['Age'],
        'Payment Delay': data['Payment_Delay'],
        'Gender': data['Gender'],
        'Subscription Type': data['Subscription_Type'],
        'Tenure': data['Tenure'],
        'Contract Length': data['Contract_Length'],
        'Usage Frequency': data['Usage_Frequency'],
        'Total Spend': data['Total_Spend'],
        'Support Calls': data['Support_Calls'],
        'Last Interaction': data['Last_Interaction']
    }

    # Define the correct feature order (same as training)
    features_order = [
        'Age',
        'Payment Delay',
        'Gender',
        'Subscription Type',
        'Tenure',
        'Contract Length',
        'Usage Frequency',
        'Total Spend',
        'Support Calls',
        'Last Interaction'
    ]

    X = pd.DataFrame([data_renamed])[features_order]

    if X.isnull().any().any() or not np.isfinite(X).all().all():
        raise HTTPException(status_code=400, detail="Input data contains NaN or infinite values.")

    try:
        X_scaled = scaler.transform(X)
        return X_scaled
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Scaling failed: {str(e)}")

@app.post("/predict")
async def predict_churn(data: CustomerData):
    try:
        input_data = data.dict()
        logger.info(f"Received input data: {input_data}")
        input_data_scaled = preprocess_data(input_data)

        model_key = data.model_type
        if model_key not in models:
            raise HTTPException(status_code=400, detail=f"Invalid model type '{model_key}'")

        model = models[model_key]
        X_scaled = input_data_scaled.reshape(1, -1)

        logger.info(f"Reshaped X_scaled shape: {X_scaled.shape}")
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        logger.info(f"Prediction: {prediction}, Probability: {probability}")

        return {
            "churn": bool(prediction),
            "probability": float(probability)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
