# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
import pickle
from mlflow.client import MlflowClient
import mlflow

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv('DAGSHUB_PAT')
        if not dagshub_token:
            raise ValueError("DAGSHUB_PAT environment variable not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/cars_ml_test.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_processed.csv')

    @staticmethod
    def get_latest_model_version(model_name):
        client = MlflowClient()
        latest_version = client.get_model_version_by_alias(model_name,'staging' )
        return latest_version.version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.drop(columns=['Price'])
        y_holdout = self.holdout_data['Price'].copy()

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        mae = mean_absolute_error(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_mae = 0.5

        # Assert that the new model meets the performance thresholds
        self.assertLessEqual(mae, expected_mae, f'Accuracy should be at least {expected_mae}')

if __name__ == "__main__":
    unittest.main()