
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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




    

    
