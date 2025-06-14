import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

# Define your model metadata for ZenML
model = Model(
    name="airport_congestion_xgb_classifier",
    license="Apache 2.0",
    description="XGBoost model for airport congestion classification."
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model,enable_step_logs=False)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="xgb_classifier_pipeline", is_model_artifact=True)]:
    """
    Builds and trains an XGBoost Classifier model using a scikit-learn pipeline.

    Parameters:
    - X_train (pd.DataFrame): Training features
    - y_train (pd.Series): Training labels

    Returns:
    - Pipeline: Trained sklearn pipeline with preprocessing + XGBoost classifier
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Feature separation
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Preprocessing steps
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    # XGBoost Classifier pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        logging.info("Training XGBoost classifier...")
        pipeline.fit(X_train, y_train)
        logging.info("Training completed.")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        mlflow.end_run()

    return pipeline
