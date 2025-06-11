import pandas as pd
class load_data:
    def __init__(self, data_source:str):
        """
        Initializing the class with the data source.
        """
        self.data_source = data_source
        

    def load(self)->pd.DataFrame:
        """
        Returning a pandas DataFrame from the data source.
        """
        return pd.read_csv(self.data_source)

        
        