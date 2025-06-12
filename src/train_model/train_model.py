
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
class Train_Model:

    def __init__(self,df):
        self.df=df

    def TestTrainSplit(self):
        data_frame=self.df.copy()
        data_frame=data_frame.sort_values(['Origin','time_window'])
        split_point = int(0.8 * len(data_frame))
        train_df = data_frame.iloc[:split_point]
        test_df = data_frame.iloc[split_point:]
        features = ['num_flights', 'avg_taxiout', 'hour', 'Day_Of_Week', 'prev_congestion', 'rolling_1h_avg_taxiout']
        target = 'congestion_label'

        X_train = train_df[features]
        y_train = train_df[target]

        X_test = test_df[features]
        y_test = test_df[target]

        return X_train,y_train,X_test,y_test
    
    def trainLogisticReg(self, X_train, y_train, X_test, y_test):
        model=LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def trainRandomForest(self, X_train, y_train, X_test, y_test):
        model=RandomForestClassifier(n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        importances = model.feature_importances_
        feature_names = X_train.columns

        # Combine into a DataFrame
        feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

        # Plot
        plt.figure(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp)
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.show()

        return y_pred

    def bestParamsRandomForest(self,X_train, y_train, X_test, y_test):
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 4],
            'max_features': ['sqrt'],
            'class_weight': ['balanced']
        }
        grid=GridSearchCV(RandomForestClassifier(),param_grid,cv=3, scoring="f1")
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nOptimized params:")
        print(grid.best_params_)

    def trainXGB(self,X_train, y_train, X_test, y_test):
        model = XGBClassifier( eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))





    

    
