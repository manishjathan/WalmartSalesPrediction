
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm
import pickle
import pandas as pd

sampleFile = open("./Data/weights.pkl","rb")
weights = pickle.load(sampleFile)
weights = np.array(weights)
sampleFile.close()

def wmae(test_weights,y_true,y_pred):
    sum_of_errors = 0
    for i in range(len(y_true)):
        sum_of_errors += test_weights[i] * (np.abs(y_true[i] - y_pred[i]))

    sum_of_weights = np.sum(np.array(test_weights))
    wmae = sum_of_errors / sum_of_weights
    return (wmae)

def loopBasedGCV(n_estimators, max_depth, X, y):

    hyper_param_df = pd.DataFrame(columns={'n_estimators','max_depth','score'})
    for n_estimator in n_estimators:
        for depth in max_depth:
            kfold = KFold(n_splits=3)
            avg_score = 0

            for train_index, test_index in kfold.split(X):

                ## Data Preparation for each fold
                Xtrain,Xtest = X.iloc[train_index], X.iloc[test_index]
                ytrain,ytest = y[train_index], y[test_index]
                test_weights = weights[test_index]

                ## RandomForest Regressor
                rfRegressor = RandomForestRegressor(n_estimators=n_estimator,max_depth=depth)
                rfRegressor.fit(Xtrain, ytrain)
                ypred = rfRegressor.predict(Xtest)


                score = wmae(test_weights, ytest, ypred)
                avg_score += score
            avg_score /= kfold.get_n_splits()

            print("N Estimators : ", n_estimator,
                  " Max Depth : ", depth,
                  " Score : ", score)

            hyper_param_df.append({'n_estimators' :n_estimator,
                                    'max_depth' : depth,
                                    'score' : avg_score},ignore_index=True)

            pickle.dump(hyper_param_df, open("lpb_hyper_param_df", "wb"))


