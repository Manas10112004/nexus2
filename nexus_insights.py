import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class InsightModule:
    """
    Advanced Analytics Module for Nexus AI.
    Features: Anomaly Detection, Forecasting, Smart Correlations.
    """

    def __init__(self):
        pass

    def check_anomalies(self, df, column_name, contamination=0.05):
        """
        Detects anomalies in a numeric column using Isolation Forest.
        Returns: A plot and a dataframe with anomalies highlighted.
        """
        # Prepare Data
        data = df[[column_name]].dropna()

        # Train Model
        model = IsolationForest(contamination=contamination, random_state=42)
        data['anomaly'] = model.fit_predict(data[[column_name]])

        # -1 indicates anomaly, 1 indicates normal
        anomalies = data[data['anomaly'] == -1]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[column_name], color='blue', label='Normal', alpha=0.6)
        plt.scatter(anomalies.index, anomalies[column_name], color='red', label='Anomaly', s=50)
        plt.title(f"Anomaly Detection: {column_name}")
        plt.legend()

        return f"Found {len(anomalies)} anomalies in '{column_name}'. (See plot above)"

    def forecast_series(self, df, date_col, value_col, periods=30):
        """
        Forecasts future values using Exponential Smoothing (Holt-Winters).
        """
        # Prep Data
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
        temp_df = temp_df.sort_values(by=date_col).set_index(date_col)

        # Resample to daily/monthly if needed (Auto-detect frequency is hard, assuming daily/index)
        # For simplicity in this demo, we assume the index is clean or we use values directly
        series = temp_df[value_col].dropna()

        if len(series) < 10:
            return "❌ Not enough data points to forecast (Need at least 10)."

        # Train Model
        try:
            model = ExponentialSmoothing(series, seasonal_periods=None, trend='add', seasonal=None).fit()
            forecast = model.forecast(periods)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(series.index, series, label='Historical')
            plt.plot(forecast.index, forecast, label='Forecast', color='green', linestyle='--')
            plt.title(f"Forecast: {value_col} ({periods} steps)")
            plt.legend()

            return f"Forecast generated for next {periods} periods. (See plot above)"
        except Exception as e:
            return f"❌ Forecasting Error: {str(e)}"

    def get_correlation_drivers(self, df, target_col):
        """
        Finds which columns correspond most strongly with the target.
        """
        numeric_df = df.select_dtypes(include=['number'])
        if target_col not in numeric_df.columns:
            return "Target column must be numeric."

        corr = numeric_df.corr()[target_col].sort_values(ascending=False)

        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(x=corr.values, y=corr.index, palette="coolwarm")
        plt.title(f"Correlation Drivers for '{target_col}'")
        plt.axvline(0, color='black', linewidth=1)

        return corr.to_string()