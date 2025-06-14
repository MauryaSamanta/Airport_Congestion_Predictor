from steps.load_data_step import load_data_step
from steps.data_processing_step import data_process_step
from steps.train_model_step import train_model_step
from steps.build_model import model_building_step

from zenml import Model, pipeline, step
import os

csv_path = os.path.abspath("data/raw/flight.csv")




@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
    enable_pipeline_logs=False
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = load_data_step(
        file_path=csv_path
    )

   

    # Feature Engineering Step
    processed_data = data_process_step(
        raw_data
    )

   
    # Data Splitting Step
    X_train, X_test, y_train, y_test = train_model_step(processed_data)

    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)

   

    return model


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
