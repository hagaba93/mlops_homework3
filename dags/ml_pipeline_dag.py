# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime
# import sys, os

# # Add src to path so DAGs can import ml_pipeline
# sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# from ml_pipeline.data import generate_data, load_data
# from ml_pipeline.model import train_model

# default_args = {"owner": "airflow", "retries": 1}

# with DAG(
#     dag_id="ml_pipeline",
#     default_args=default_args,
#     description="Pipeline: generate data -> train model",
#     schedule_interval=None,
#     start_date=datetime(2025, 1, 1),
#     catchup=False,
# ) as dag:

#     generate_task = PythonOperator(
#         task_id="generate_data",
#         python_callable=generate_data,
#         # op_kwargs={"output_path": "data/iris.csv"},
#         op_kwargs={"output_path": "data/breast_cancer.csv"},
#     )

#     def train_model_wrapper(data_path: str, model_path: str):
#         df = load_data(data_path)
#         return train_model(df, model_path)

#     train_task = PythonOperator(
#         task_id="train_model",
#         python_callable=train_model_wrapper,
#         op_kwargs={
#             # "data_path": "data/iris.csv",
#             # "model_path": "models/iris_model.pkl",
#             "data_path": "data/breast_cancer.csv",
#             "model_path": "models/breast_cancer_model.pkl",
#         },
#     )

#     generate_task >> train_task

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import json
import os
import pickle

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import boto3

MODEL_DIR = "/home/ec2-user/environment/lab4_model_training/models"
# BUCKET_NAME = "mlops-2026-spring"
BUCKET_NAME = "agabs3mlops"
ACCURACY_THRESHOLD = 0.95



# Train
def train_model(**context):
    """Load breast cancer data.Train a logistic regression classifier and save it."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data and split
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Train
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    import joblib
    joblib.dump(model, os.path.join(MODEL_DIR, "breast_cancer_model.pkl"))

    # Save test data so evaluate_model can use it
    with open(os.path.join(MODEL_DIR, "test_data.pkl"), "wb") as f:
        pickle.dump({"X_test": X_test, "y_test": y_test}, f)

    # Generate version from execution date and push
    model_version = context["execution_date"].strftime("%Y%m%d_%H%M%S")
    context["ti"].xcom_push(key="model_version", value=model_version)

    print(f"Model trained and saved. Version: {model_version}")



# Evaluate
def evaluate_model(**context):
    """Compute accuracy, save metrics.json and metadata.json."""
    import joblib

    # Load model
    model = joblib.load(os.path.join(MODEL_DIR, "breast_cancer_model.pkl"))

    # Load test data
    with open(os.path.join(MODEL_DIR, "test_data.pkl"), "rb") as f:
        test_data = pickle.load(f)

    # Evaluate
    predictions = model.predict(test_data["X_test"])
    accuracy = round(accuracy_score(test_data["y_test"], predictions), 4)

    # Get version from train step
    model_version = context["ti"].xcom_pull(
        task_ids="train_model", key="model_version"
    )

    # Save metrics.json
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)

    # Save metadata.json
    metadata = {
        "model_version": model_version,
        "dataset": "breast_cancer",
        "model_type": "logistic_regression",
        "accuracy": accuracy,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Push accuracy so promote step can check it
    context["ti"].xcom_push(key="accuracy", value=accuracy)

    print(f"Evaluation complete -- accuracy: {accuracy}")


#Promote
def promote_model(**context):
    """Upload artifacts to S3 if accuracy meets threshold"""
    accuracy = context["ti"].xcom_pull(task_ids="evaluate_model", key="accuracy")
    model_version = context["ti"].xcom_pull(
        task_ids="train_model", key="model_version"
    )

    print(f"Accuracy: {accuracy} | Threshold: {ACCURACY_THRESHOLD}")

    # No pass if not
    if accuracy < ACCURACY_THRESHOLD:
        raise ValueError(
            f"Model accuracy {accuracy} is below threshold {ACCURACY_THRESHOLD}. "
            "Promotion aborted."
        )

    # Upload to S3
    s3 = boto3.client("s3")
    prefix = f"models/{model_version}"

    for filename in ["breast_cancer_model.pkl", "metrics.json", "metadata.json"]:
        local_path = os.path.join(MODEL_DIR, filename)
        s3_key = f"{prefix}/{filename}"
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        print(f"Uploaded {local_path} -> s3://{BUCKET_NAME}/{s3_key}")

    print(f"Model {model_version} promoted successfully!")


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="Train -> Evaluate -> Promote breast cancer model to S3",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    train = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    promote = PythonOperator(task_id="promote_model", python_callable=promote_model)

    train >> evaluate >> promote