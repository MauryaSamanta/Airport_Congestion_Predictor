import pandas as pd
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

  