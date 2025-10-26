from fastapi import FastAPI
from mlflow.client import MlflowClient
from pydantic import BaseModel, StrictInt
import mlflow
import dagshub
import pandas as pd

app = FastAPI()

mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/cars_ml_test.mlflow')
dagshub.init(repo_owner='akshatsharma2407', repo_name='cars_ml_test', mlflow=True)

class InputSchema(BaseModel):
    Model_Year: float
    Mileage: float
    Accidents_Or_Damage : float
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

def get_latest_model_version(model_name):
    client = MlflowClient()
    latest_version = client.get_model_version_by_alias(model_name,'staging' )
    if not latest_version:
        latest_version = client.get_model_version_by_alias(model_name, 'None')
    return latest_version.version if latest_version else None

model_name = 'cars_model'

model_version = get_latest_model_version(model_name)

model_uri = f"models:/{model_name}/{model_version}" 
model = mlflow.pyfunc.load_model(model_uri)

@app.post('/predict')
def prediction(user_input: InputSchema):
    input_df = pd.DataFrame([user_input.model_dump()])  # convert input to DataFrame
    prediction = model.predict(input_df)                # ML model prediction
    return {"prediction": prediction.tolist()}          # convert to list for JSON
