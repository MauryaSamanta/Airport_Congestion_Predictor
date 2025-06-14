import pandas as pd
from src.data_processing.process import Process
from zenml import step

@step(enable_step_logs=False)
def data_process_step(raw_data:pd.DataFrame)->pd.DataFrame:
    Processor=Process(raw_data)
    delayed_column_added=Processor.addColumnWhetherDelayed()
    time_standard_df=Processor.standardizeTime()
    dropped_columns_df=Processor.dropColumnNulls()
    all_dates_to_time_df=Processor.convertTimetoDateTime()
    taxiOut_thresholds=Processor.getairportthresholds()
    df_with_congestion_label=Processor.label_congestion_column(taxiout_thresholds=taxiOut_thresholds)
    df_with_time_windows=Processor.addTimeWindowColumn()
    time_window_df=Processor.createTimeWindowSet(df=df_with_time_windows)
    new_features_df=Processor.createNewFeatures(df=time_window_df)
    return new_features_df


