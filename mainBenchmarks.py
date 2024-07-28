# region Import packages

import statsmodels.api as sm
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.regression.quantile_regression import QuantReg

# select benchmark model
ARX_MODEL = True
QUANTILE_REGRESSION = False
TAO_MODEL = False

# importing dataset
df = pd.read_csv("final_dataset.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# selecting time intervals
start_date = pd.to_datetime("2008-01-01", format="%Y-%m-%d")
mid_date = pd.to_datetime("2012-01-01", format="%Y-%m-%d")
end_date = pd.to_datetime("2013-01-01", format="%Y-%m-%d")

# selecting training set and test set
df_train = df[(df.index >= start_date) & (df.index < mid_date)]
df_test = df[(df.index >= mid_date) & (df.index < end_date)]

# define regressand variable
y_train = df_train["Demand"].copy()
y_test = df_test["Demand"].copy()

# define collectors Prediction Intervals
bipercentiles = np.zeros(shape=(len(y_test), 199))
coverage = np.zeros(shape=10)
PL = np.zeros(shape=(99, 1))

# ARX Model
if ARX_MODEL:

    regressors = ["Intercept", "Trend",
                  "Drybulb", "Dewpnt",
                  "SY1", "CY1", "SY2", "CY2",
                  "SD1", "CD1", "SD2", "CD2",
                  "DoW_0","DoW_1", "DoW_2", "DoW_3", "DoW_4",  # (Mon, ..., Fri)
                  "DoW_5", "DoW_6", "Holiday"]  # (Sat, Sun, Hol)

    # generating regressors and targets
    x_train = df_train[regressors].copy()
    x_test = df_test[regressors].copy()

    # additional variables, in both x_train and x_test
    x_train["Drybulb2"], x_train["Drybulb3"] = x_train["Drybulb"] ** 2, x_train["Drybulb"] ** 3
    x_train["Dewpnt2"],  x_train["Dewpnt3"]  = x_train["Dewpnt"] ** 2,  x_train["Dewpnt"] ** 3

    x_test["Drybulb2"], x_test["Drybulb3"] = x_test["Drybulb"] ** 2, x_test["Drybulb"] ** 3
    x_test["Dewpnt2"],  x_test["Dewpnt3"]  = x_test["Dewpnt"] ** 2,  x_test["Dewpnt"] ** 3

    # create model
    model = AutoReg(y_train, lags=[1, 2, 24], exog=x_train).fit()
    print(model.ar_lags)
    model = model.model.fit()

    # make point forecasts
    point_forecast = model.predict(start=x_test.index[0], end=x_test.index[-1],  exog_oos=x_test)

    # generate PIs and fill bipercentiles
    for idx in range(1, 100):
        one_minus_alpha = idx/100
        tmp = model.get_prediction(start=x_test.index[0], end=x_test.index[-1],  exog_oos=x_test).conf_int(alpha=one_minus_alpha)
        bipercentiles[:, idx-1] = np.array(tmp['lower'])
        bipercentiles[:, -idx] = np.array(tmp['upper'])


# Tao Model
if TAO_MODEL:

    x_train = df_train[["Intercept", "Trend"]].copy()
    # dummy sets (month-of-year, hour-of-day, hour-of-week)
    moy_set = pd.get_dummies(x_train.index.month, prefix="M").set_index(x_train.index)
    hod_set = pd.get_dummies(x_train.index.hour, prefix="H").set_index(x_train.index)
    how_vector = x_train.index.dayofweek*24 + x_train.index.hour
    how_set = pd.get_dummies(how_vector, prefix="HoW").set_index(x_train.index)

    # interactions with temperature
    T = df_train["Drybulb"]
    T1H, T2H, T3H = hod_set.multiply(T, axis=0), hod_set.multiply(T**2, axis=0), hod_set.multiply(T**3, axis=0)
    T1M, T2M, T3M = moy_set.multiply(T, axis=0), moy_set.multiply(T**2, axis=0), moy_set.multiply(T**3, axis=0)

    x_train = pd.concat((x_train, moy_set, how_set, T1H, T2H, T3H, T1M, T2M, T3M), axis=1)

    x_test = df_test[["Intercept", "Trend"]].copy()
    # dummy sets (month-of-year, hour-of-day, hour-of-week)
    moy_set = pd.get_dummies(x_test.index.month, prefix="M").set_index(x_test.index)
    hod_set = pd.get_dummies(x_test.index.hour, prefix="H").set_index(x_test.index)
    how_vector = x_test.index.dayofweek * 24 + x_test.index.hour
    how_set = pd.get_dummies(how_vector, prefix="HoW").set_index(x_test.index)

    # interactions with temperature
    T = df_test["Drybulb"]
    T1H, T2H, T3H = hod_set.multiply(T, axis=0), hod_set.multiply(T ** 2, axis=0), hod_set.multiply(T ** 3, axis=0)
    T1M, T2M, T3M = moy_set.multiply(T, axis=0), moy_set.multiply(T ** 2, axis=0), moy_set.multiply(T ** 3, axis=0)

    x_test = pd.concat((x_test, moy_set, how_set, T1H, T2H, T3H, T1M, T2M, T3M), axis=1)

    # create model
    model = sm.OLS(np.array(y_train, dtype=float), np.array(x_train, dtype=float)).fit()

    # make point forecasts
    point_forecast = model.predict(np.array(x_test, dtype=float))

    # generate PIs and fill bipercentiles
    for idx in range(1, 100):
        one_minus_alpha = idx/100
        tmp = model.get_prediction(np.array(x_test, dtype=float)).summary_frame(alpha=one_minus_alpha)
        bipercentiles[:, idx-1] = tmp.obs_ci_lower
        bipercentiles[:, -idx] = tmp.obs_ci_upper


# define regressors for QR
if QUANTILE_REGRESSION:

    regressors = ["Intercept", "Trend",
                  "Drybulb", "Dewpnt",
                  "SY1", "CY1", "SY2", "CY2",
                  "SD1", "CD1", "SD2", "CD2",
                  "DoW_0","DoW_1", "DoW_2", "DoW_3", "DoW_4",  # (Mon, ..., Fri)
                  "DoW_5", "DoW_6", "Holiday"]  # (Sat, Sun, Hol)

    # generating regressors and targets
    x_train = df_train[regressors].copy()
    x_test = df_test[regressors].copy()

    # additional variables, in both x_train and x_test
    x_train["Drybulb2"], x_train["Drybulb3"] = x_train["Drybulb"] ** 2, x_train["Drybulb"] ** 3
    x_train["Dewpnt2"],  x_train["Dewpnt3"]  = x_train["Dewpnt"] ** 2,  x_train["Dewpnt"] ** 3

    x_test["Drybulb2"], x_test["Drybulb3"] = x_test["Drybulb"] ** 2, x_test["Drybulb"] ** 3
    x_test["Dewpnt2"],  x_test["Dewpnt3"]  = x_test["Dewpnt"] ** 2,  x_test["Dewpnt"] ** 3

    # create quantile forecasts
    for idx in tqdm(range(1, 200), "Performing Quantile Regression"):
        alpha = idx/200
        qr = QuantReg(np.log(y_train), x_train)
        pred_alpha = qr.fit(q=alpha, max_iter=2000).predict(x_test)
        bipercentiles[:, idx-1] = np.exp(pred_alpha.copy())

    # compute point forecasts as the average of quantiles (using trapezoidal rule on bipercentiles)
    point_forecast = (bipercentiles.sum(axis=1) - 0.5 * (bipercentiles[:, 0] + bipercentiles[:, -1])) / 198
    # and store median forecast
    median_forecast = bipercentiles[:, 99]


# compute pinball loss on percentiles
for idx in range(1, 100):
    alpha = idx/100
    percentile = bipercentiles[:, 2*idx-1].copy()
    PL[idx-1] = np.mean(alpha * np.maximum(y_test-percentile, 0) + (1-alpha) * np.maximum(percentile-y_test, 0))

# compute coverage of PI on high confidence levels
for idx in range(90, 100):
    alpha = idx/100
    PI_dw, PI_up = bipercentiles[:, 99-idx], bipercentiles[:, 99+idx]
    coverage[idx-90] = np.round(np.mean((y_test >= PI_dw) * (y_test <= PI_up)), 5)


# computing metrics on point forecast (expected value)
MAPE = np.mean(np.abs(point_forecast/y_test - 1)) * 100
RMSE = np.sqrt(np.mean((point_forecast-y_test)**2))

# computing average Pinball Loss
APL = np.mean(PL)

# compute AACE
AbsoluteCoverageError = np.abs(100*coverage - np.arange(90,100))
AACE = np.mean(AbsoluteCoverageError[AbsoluteCoverageError < 30])  # remove diverged PIs

# outputs
print("MAPE:", np.round(MAPE, 2))
print("RMSE:", np.round(RMSE, 2))
print("APL:", np.round(APL, 2))
print("AACE:", np.round(AACE, 2))
print("Coverage[90]:", np.round(100*coverage[0], 2))
print("Coverage[95]:", np.round(100*coverage[5], 2))
