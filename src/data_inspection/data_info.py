import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class data_info:
    def __init__(self, dataframe: pd.DataFrame):
        self.df=dataframe

    def get_info(self):
        """
        Returns the information of the DataFrame.
        """
        print("\nData Types and Non-null Counts:")
        print(self.df.info())

    def get_stat_summary(self):
        """
        Return the statistical summary of the DataFrame.
        """
        print(self.df.describe())

    def error_analysis(self,X_test, y_test, y_pred):
        results = X_test.copy()
        results['actual'] = y_test.values
        results['predicted'] = y_pred

        # Filter where actual = 0 and predicted = 1
        false_positives = results[(results['actual'] == 0) & (results['predicted'] == 1)]

        print(false_positives.head())

    def graph_analysis_error(self,X_test, y_test, y_pred, feature):
        results = X_test.copy()
        results['actual'] = y_test.values
        results['predicted'] = y_pred
        # False positives (actual=0, predicted=1)
        fp = results[(results['actual'] == 0) & (results['predicted'] == 1)]

        # True negatives (actual=0, predicted=0)
        tn = results[(results['actual'] == 0) & (results['predicted'] == 0)]

        #   Example feature to compare (e.g., avg_taxiout)
        sns.histplot(fp[feature], color='red', label='False Positives', kde=True)
        sns.histplot(tn[feature], color='green', label='True Negatives', kde=True)
        plt.legend()
        plt.title(f'{feature} distribution: FP vs TN')
        plt.show()
