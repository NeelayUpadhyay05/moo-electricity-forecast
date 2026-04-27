from statsmodels.tsa.arima.model import ARIMA
import numpy as np


def fit_arima(train_series, order=(5, 1, 0)):
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    return fitted


def forecast_arima(fitted_model, steps):
    pred = fitted_model.forecast(steps=steps)
    return np.asarray(pred)
