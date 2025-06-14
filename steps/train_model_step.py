import pandas as pd
from src.train_model.train_model import Train_Model
from zenml import step
from typing import Tuple
@step(enable_step_logs=False)
def train_model_step(cleaned_data)-> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    ModelTrainer=Train_Model(cleaned_data)
    print(ModelTrainer.TestTrainSplit())
    X_train, y_train, X_test, y_test=ModelTrainer.TestTrainSplit()
    return X_train, y_train, X_test, y_test