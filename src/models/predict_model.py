import json
import joblib
import numpy as np
import logging
import os
import yaml
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from dvclive import Live
import mlflow
import seaborn as sns
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/cars_ml_test.mlflow')

dagshub.init(repo_owner='akshatsharma2407', repo_name='cars_ml_test', mlflow=True)

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel('DEBUG')

file_handler = logging.FileHandler('reports/errors.log')
file_handler.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def load_model(model_path: str) -> BaseEstimator:
    try:
        model = joblib.load(model_path)
        logger.debug('ml model loaded')
        return model
    except FileNotFoundError:
        logger.error(f'File not found at given location {__file__} -> {model_path}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error at {__file__} -> load_model')
        raise

def load_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        test_data = pd.read_csv(data_path)

        xtest = test_data.drop(columns="Price")
        ytest = test_data["Price"]
        logger.debug('data loaded')
        return xtest, ytest
    except FileNotFoundError:
        logger.error(f'File not found at given location {__file__} -> {data_path}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error at {__file__} -> load_data')
        raise


def evaluate(
    xtest: pd.DataFrame,
    ytest: pd.DataFrame,
    model: BaseEstimator,
    evaluation_result_path: str,
) -> float:
    try:
        ypred = model.predict(xtest.values)

        mae = mean_absolute_error(ytest.values, ypred)

        metrics_dict = {"MAE": mae}

        with open(evaluation_result_path, "w") as f:
            json.dump(metrics_dict, f)
        
        logger.debug('evaluation done')

        return mae
    except Exception as e:
        logger.error(f'found unexpected error at {__file__} -> evaluate')
        raise

def saved_run_id(run_id : str,model_name: str, path: str) -> None:
    try:
        model_info = {'run_id': run_id, 'model_name': model_name}
        with open(path, 'w') as file:
            json.dump(model_info, file)
    except Exception as e:
        logger.error('Error occured while saving the model')
        raise
    

def exp_tracking_mlflow(params_path: str, mae: float, xtest: pd.DataFrame, model: BaseEstimator) -> None:
    mlflow.sklearn.autolog()
    mlflow.set_experiment('model eval before registering')
    with mlflow.start_run() as run:

        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        
        for param, value in params.items():
            for key, val in value.items():
                mlflow.log_param(f'{param}_{key}',val)
        
        corr_matrix = xtest.corr()
        sns.heatmap(corr_matrix)
        
        plt.savefig('corr_matrix.png')
        mlflow.log_artifact('corr_matrix.png')

        mlflow.log_artifact(__file__)

        signature = mlflow.models.infer_signature(model_input=xtest.head(),model_output=model.predict(xtest.head(2)))
        model_name = 'model'
        mlflow.sklearn.log_model(model,model_name, signature=signature)

        saved_run_id(run.info.run_id,model_name,'reports/run_info.json')

        mlflow.set_tag('author','akshat') 

def main() -> None:
    try:
        model = load_model(model_path="models/model.pkl")
        xtest, ytest = load_data(data_path="data/processed/test_processed.csv")
        mae = evaluate(
            xtest=xtest,
            ytest=ytest,
            model=model,
            evaluation_result_path="reports/metrics.json"
        )
        exp_tracking_mlflow(params_path='params.yaml',mae=mae, xtest= xtest, model= model)
        logger.debug('main function executed')
    except Exception as e:
        logger.error(f'Found Unexpected error at {__file__} -> main')
        raise


if __name__ == "__main__":
    main()