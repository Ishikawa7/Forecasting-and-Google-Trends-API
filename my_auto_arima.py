import pandas as pd
import numpy as np
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA

def predictions(y):
    dates = pd.date_range(start=y["Data"].iloc[-1],periods=8)[1:]
    fh = ForecastingHorizon(np.arange(y.index[-1]+1,y.index[-1] + 8),is_relative=False)
    df_pred = pd.DataFrame(index = dates)
    df_pred.index.name = "Data"
    forecaster = StatsForecastAutoARIMA(sp=7)
    forecaster.fit(y[["Ospiti"]])
    y_pred = forecaster.predict(fh)
    df_pred["AutoArima Forecasts"] = list(y_pred["Ospiti"])
    return df_pred