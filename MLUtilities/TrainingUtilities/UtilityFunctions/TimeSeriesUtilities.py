from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def rolling_forecast_generic(train_data, cv_data, trend_order, s_order, verbose=False):
    """
    train_data : training signal
    cv_data : cross_validation signal
    trend_order : Trend Order(p,d,q)
    seasonal_order : Seasonal Parameters(P,D,Q,s)
    """

    for i in range(len(cv_data)):
        sarima = SARIMAX(train_data, order=trend_order, seasonal_order=s_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        sarima_mdl = sarima.fit()
        pred_date = cv_data.index[i]  # referring to the date in cv_data
        pred = sarima_mdl.predict(start=pred_date, end=pred_date)  # Predict the next forecast
        train_data[pred_date] = np.float(pred[0])  # Append the prediction data to train data
        if verbose == True:
            print("*" * 30)
            print("pred date : ", pred_date)
            print("Prediction : ", pred[0])
            print("Last point added to train data : ", train_data[-1])

    return train_data[-len(cv_data):]