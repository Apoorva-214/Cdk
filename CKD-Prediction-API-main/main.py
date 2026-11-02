from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import time


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="CKD Prediction API",
    description="REST API for Chronic Kidney Disease prediction",
    version="1.0.0"
)


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

model = joblib.load("model/ckd_model.joblib")


# --------------------------------------------------
# 3. API metrics
# --------------------------------------------------

total_requests = 0
successful_predictions = 0
failed_requests = 0
total_response_time = 0.0


# --------------------------------------------------
# 4. Input data schema
# --------------------------------------------------

class CKDInput(BaseModel):

    age: float = Field(..., ge=0, le=120)
    bp: float = Field(..., ge=0)
    sg: float = Field(..., ge=0)
    al: float = Field(..., ge=0)
    su: float = Field(..., ge=0)

    rbc: str
    pc: str
    pcc: str
    ba: str

    bgr: float = Field(..., ge=0)
    bu: float = Field(..., ge=0)
    sc: float = Field(..., ge=0)
    sod: float = Field(..., ge=0)
    pot: float = Field(..., ge=0)
    hemo: float = Field(..., ge=0)
    pcv: float = Field(..., ge=0)
    wbcc: float = Field(..., ge=0)
    rbcc: float = Field(..., ge=0)

    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str


# --------------------------------------------------
# 5. GET /
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "CKD Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


# --------------------------------------------------
# 6. GET /health
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# 7. GET /ready
# --------------------------------------------------

@app.get("/ready")
def ready():

    if model is not None:
        return {
            "ready": True,
            "model_loaded": True
        }

    return {
        "ready": False,
        "model_loaded": False
    }


# --------------------------------------------------
# 8. GET /model-info
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    return {
        "model_name": "CKD Classifier",
        "version": "1.0.0",
        "algorithm": "Random Forest Classifier",
        "number_of_features": 24,
        "accuracy": 1.0000,
        "precision": 1.0000,
        "recall": 1.0000,
        "f1_score": 1.0000
    }


# --------------------------------------------------
# 9. POST /predict
# --------------------------------------------------

@app.post("/predict")
def predict(data: CKDInput):

    global total_requests
    global successful_predictions
    global failed_requests
    global total_response_time

    start_time = time.time()

    total_requests += 1

    try:

        # Convert input to dictionary
        input_data = data.model_dump()

        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Get probability
        probabilities = model.predict_proba(input_df)[0]

        ckd_probability = float(probabilities[1])

        # Convert prediction to readable output
        if prediction == 1:
            result = "CKD"
        else:
            result = "Not CKD"

        successful_predictions += 1

        response_time = time.time() - start_time
        total_response_time += response_time

        return {
            "prediction": result,
            "model_probability": round(ckd_probability, 4),
            "model_version": "1.0.0",
            "message": "Model output only; not a medical diagnosis."
        }

    except Exception as e:

        failed_requests += 1

        return {
            "error": "Prediction failed",
            "details": str(e)
        }


# --------------------------------------------------
# 10. GET /metrics
# --------------------------------------------------

@app.get("/metrics")
def metrics():

    average_response_time = 0

    if successful_predictions > 0:
        average_response_time = (
            total_response_time / successful_predictions
        )

    return {
        "total_requests": total_requests,
        "successful_predictions": successful_predictions,
        "failed_requests": failed_requests,
        "average_response_time_ms": round(
            average_response_time * 1000,
            2
        )
    }