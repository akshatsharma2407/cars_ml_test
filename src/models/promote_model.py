# promote model

import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/cars_ml_test.mlflow')

    client = mlflow.MlflowClient()

    model_name = "cars_model"
    
    latest_version_staging = client.get_model_version_by_alias(model_name, 'staging').version

    # Archive the current production model
    try:
        prod_version = client.get_model_version_by_alias(model_name, "Production")
    except:
        prod_version = None

    if prod_version:
        client.set_registered_model_alias(
                name=model_name,
                version=prod_version.version,
                stage="Archived"
            )

    # Promote the new model to production
    client.set_registered_model_alias(
        name=model_name,
        version=latest_version_staging,
        alias="Production"
    )
    
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()