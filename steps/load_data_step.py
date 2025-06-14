import pandas as pd
from src.data_ingestion.load_data import load_data
from zenml import step

@step(enable_step_logs=False)
def load_data_step(file_path:str)->pd.DataFrame:
    DataLoader=load_data(data_source=file_path)
    df=DataLoader.load()
    return df