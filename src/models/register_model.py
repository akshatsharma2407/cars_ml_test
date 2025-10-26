from mlflow.tracking import MlflowClient
import mlflow
import json

mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/cars_ml_test.mlflow')

client = MlflowClient(tracking_uri='https://dagshub.com/akshatsharma2407/cars_ml_test.mlflow')

with open('reports/run_info.json') as f:
    run_info = json.load(f)

model_uri = f'runs:/{run_info[['run_id']]}/{run_info['model_name']}'

result = mlflow.register_model(model_uri=model_uri,name='cars_model')

client.update_model_version(
    name='cars_model',
    version=result.version,
    description='a new version of model for production'
)

client.set_registered_model_alias(
    name='cars_model',
    alias='staging',
    version=result.version
)

client.set_model_version_tag(
    name='cars_model',
    version=result.version,
    key='new version',
    value='before testing'
)