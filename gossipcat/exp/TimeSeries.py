#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def get_procedure():
    print('Modelling procedure:\n \
            1. Plot the data. Identify any unusual observations.\n \
            2. If necessary, transform the data (using a Box-Cox transformation) to stabilize the variance.\n \
            3. If the data are non-stationary: take first differences of the data until the data are stationary.\n \
            4. Examine the ACF/PACF: Is an AR(p) or MA(q) model appropriate?\n \
            5. Try your chosen model(s), and use the AICc to search for a better model.\n \
            6. Check the residuals from your chosen model by plotting the ACF of the residuals, and doing a portmanteau test of the residuals. If they do not look like white noise, try a modified model.\n \
            7. Once the residuals look like white noise, calculate forecasts.\n')

def trans_time(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col])
    return df

def plot_acf_pacf(ts):
    plot_acf(ts)
    plt.figure(0)
    plot_pacf(ts)
    plt.show()

def diagnostic(df, col, order=(0, 0, 0), test_size=0.2):
    ts = df[col].dropna().values

    model = ARIMA(ts, order=order)
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.figure(0)
    residuals.plot(kind='kde')
    plt.figure(0)
    print(residuals.describe())

    # calculate MSE wity sliding windows
    size = int(len(ts) * (1 - test_size))

    train, test = ts[0:size], ts[size:len(ts)]
    history = [x for x in train]

    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    
    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()
