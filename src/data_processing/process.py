import pandas as pd
class Process:
    def __init__(self, df):
        self.df=df
    
    def addColumnWhetherDelayed(self):
        """
        Here adding a column of 1 or 0 according to whether the flight is delayed or not
        """
        data_frame=self.df.copy()
        data_frame['is_delayed']=data_frame['CarrierDelay'].apply(lambda x: 1 if x>20 else 0)
        """
        We took the value 15 because FAA considers 15minutes as the threshold for a flight to be considered delayed.
        """
        return data_frame
    
    def standardizeTime(self):
        """
        Here standardizing the time columns to 24-hour format
        """
        data_frame=self.df.copy()
        data_frame['DepTime'] = pd.to_datetime(data_frame['DepTime'], format='%H%M', errors='coerce').dt.strftime('%H:%M')
        data_frame['ArrTime'] = pd.to_datetime(data_frame['ArrTime'], format='%H%M', errors='coerce').dt.strftime('%H:%M')
        return data_frame
    
    def dropColumnNulls(self, columns_to_drop=None):
        """
        Here dropping the columns with null values
        """
        data_frame = self.df.copy()
        data_frame.dropna(subset=columns_to_drop, inplace=True)
        return data_frame
        
    
    def convertTimetoDateTime(self):
   
        df = self.df.copy()

    # Remove nulls
        df = df[df['DepTime'].notnull()]

    # Convert to string, strip spaces, and filter only numeric
        df['DepTime'] = df['DepTime'].astype(str).str.strip()

    # Remove anything that’s not all digits
        df = df[df['DepTime'].str.match(r"^\d+$")]

    # Pad to 4 digits ONLY if <= 4 digits
        df = df[df['DepTime'].str.len() <= 4]
        df['DepTime'] = df['DepTime'].str.zfill(4)

    # Now convert to datetime safely
        df['DepTime'] = pd.to_datetime(df['DepTime'], format='%H%M', errors='coerce')

    # Drop rows where conversion failed
        df = df[df['DepTime'].notnull()]

    # Bin into 15-min intervals
        df['time_window'] = df['DepTime'].dt.floor('15min')

        return df

    def getairportthresholds(self):
        """
        Here we are computing airport-wise TaxiOut time  75th percentiles and storing them in a Pandas Series.
        """
        data_frame=self.df.copy()
        taxiout_thresholds = data_frame.groupby('Origin')['TaxiOut'].quantile(0.75)
        return taxiout_thresholds
    
    def label_congestion(row, threshold_map):
        origin = row['Origin']
        taxi_out = row['TaxiOut']
    
        threshold = threshold_map.get(origin, None)
    
        if threshold is None:
            return 0  # fallback if airport threshold not found
    
        return 1 if taxi_out > threshold else 0
    
    def label_congestion_column(self, taxiout_thresholds):
        data_frame=self.df
        data_frame['congested'] = (data_frame['TaxiOut'] > data_frame['Origin'].map(taxiout_thresholds)).astype(int)
        return data_frame
    
    def addTimeWindowColumn(self):
        data_frame = self.df.copy()
    
        # Convert to datetime safely
        data_frame['DepTime'] = pd.to_datetime(
            data_frame['DepTime'], errors='coerce', format='%H%M'
        )
    
        # Drop invalid times
        data_frame = data_frame.dropna(subset=['DepTime'])

        # Create 15-min window
        data_frame['time_window'] = data_frame['DepTime'].dt.floor('15min')
    
        return data_frame
    
    def createTimeWindowSet(self,df):
        data_frame=df.copy()
        window_df = data_frame.groupby(['Origin', 'time_window']).agg(
        num_flights=('FlightNum', 'count'),
        avg_taxiout=('TaxiOut', 'mean'),
        num_congested=('congested', 'sum'),
        Day_Of_Week=('DayOfWeek','first')
        ).reset_index()
        window_df['congestion_label']=(window_df['num_congested']>0).astype(int)
        return window_df
    
    def createNewFeatures(self, df):
        data_frame=df.copy()
        data_frame['hour']=data_frame['time_window'].dt.hour
        data_frame = data_frame.sort_values(['Origin', 'time_window'])
        data_frame['prev_congestion'] = data_frame.groupby('Origin')['congestion_label'].shift(1).fillna(0)
        data_frame['rolling_1h_avg_taxiout'] = (
        data_frame
        .groupby('Origin')['avg_taxiout']
        .transform(lambda x: x.rolling(window=4, min_periods=1).mean())  # 4 windows = 1 hour if each is 15 min
        )
        return data_frame









