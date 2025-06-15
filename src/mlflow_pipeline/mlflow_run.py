import mlflow
import joblib
class mlflow_run:
    def __init__(self):
        pass

    def start_run(self, report_dict, model):
        mlflow.end_run()
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        mlflow.set_experiment("XGBoost Experiment_V3")
        with mlflow.start_run(run_name="XGBoost Run_V3.3"):
            mlflow.log_metric("accuracy", report_dict["accuracy"])
            mlflow.log_metric("f1_score", report_dict["weighted avg"]["f1-score"])
            joblib.dump(model, "model.pkl")
            mlflow.log_artifact("model.pkl")
            

        mlflow.sklearn.log_model(model, "XGBoost Classifier")
        
