from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pickle

def calcRMSE(y_true, y_pred):
    return (np.sqrt(mse(y_true, y_pred)))

def calcWMAE(y_true, y_pred):
    weights = pickle.load(open('./Data/weights.pkl','rb'))
    sum_of_errors = 0
    for i in range(len(y_true)):
        sum_of_errors += weights[i]*(np.abs(y_true[i] - y_pred[i]))

    sum_of_weights = np.sum(np.array(weights))
    wmae = sum_of_errors/sum_of_weights

    print("Sum of errors : ", sum_of_errors,
          " Sum of weights : ", sum_of_weights,
          " WMAE : ", wmae)
    return(wmae)

