import matplotlib.pyplot as plt
import seaborn as sns
class MissingValueAnalyser:
    def __init__(self, df):
        """
        Initializing the class with the DataFrame.
        """
        self.df = df
    
    def get_analysis(self):
        print("\nMissing Values Count by Column:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])

    def get_missing_value_visualise(self):
        """
        Visualize the missing values in the DataFrame.
        """
      
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()
