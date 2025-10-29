from fastapi import FastAPI
from mlflow.client import MlflowClient
from pydantic import BaseModel
import mlflow
import os
import pandas as pd
import uvicorn

app = FastAPI()

dagshub_token = os.getenv('DAGSHUB_PAT')
if not dagshub_token:
    raise ValueError("DAGSHUB_PAT environment variable not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/cars_ml_test.mlflow')

class InputSchema(BaseModel):
    Model_Year: float
    Mileage: float
    Accidents_Or_Damage: float
    Clean_Title: float
    One_Owner_Vehicle: float
    Personal_Use_Only: float
    Level2_Charging: float
    Dc_Fast_Charging: float
    Battery_Capacity: float
    Expected_Range: float
    Gear_Spec: float
    Engine_Size: float
    Valves: float

def get_latest_model_version(model_name: str):
    client = MlflowClient()
    latest_version = client.get_model_version_by_alias(model_name, "Production")
    if not latest_version:
        latest_version = client.get_model_version_by_alias(model_name, "None")
    return latest_version.version if latest_version else None


model_name = "cars_model"
model_version = get_latest_model_version(model_name)

if model_version is None:
    raise ValueError(f"No valid model version found for '{model_name}'")

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/")
def home():
    return {'message': 'Welcome to the Car Price Prediction API'}

@app.get('/about')
def about():
    return {'author': 'Akshat Sharma', 'version': '1.0', 'description': 'API for predicting car prices using ML model'} 

@app.get('/model_version')
def model_version_info():
    return {'model_name': 'demo model', 'model_version': '2'}

@app.post("/predict")
def prediction(user_input: InputSchema):
    input_df = pd.DataFrame([user_input.model_dump()])
    prediction = model.predict(input_df)
    return {"prediction": prediction.tolist()}
