#from Director.LibBasedGCV import libBasedGCV
from Director.LoopBasedGCV import loopBasedGCV
from Director.DataPreparation import prepareTrainTestData
from TrainingUtilities.UtilityFunctions import TimeSeriesUtilities
import pickle
import pandas as pd
from EvaluationUtilities.ModelEvaluation import calcRMSE

if __name__ == '__main__':


    train_df = pd.read_csv('./Data/train.csv','rb')
    train_df['weights'] = train_df.apply(lambda row : 5 if train_df['IsHoliday'] == True else 1)

    sd11 = pickle.load(open('./Data/sd11.pkl','rb'))
    sd11_train = sd11[:100]
    sd11_cv = sd11[100:]

    SP = [i for i in range(0,3)]
    SD = [i for i in range(0,3)]
    SQ = [i for i in range(0,3)]
    SS = [i for i in range(47,53)]

    optimal_trend_param = ()
    optimal_seasonal_param = ()
    true_values = sd11_cv
    min_rmse = 1000000
    for p in SP:
        for d in SD:
            for q in SQ:
                for s in SS:
                    seasonal_order = (p,d,q,s)
                    pred_values = TimeSeriesUtilities.rolling_forecast_generic(sd11_train,sd11_cv,
                                                                 trend_order=(0,0,0),
                                                                 s_order = seasonal_order)
                    rmse = calcRMSE(true_values,pred_values)
                    if min_rmse > rmse:
                        min_rmse = rmse
                        optimal_seasonal_param = seasonal_order

    print("Optimal Seasonal Order : ",optimal_seasonal_param)

