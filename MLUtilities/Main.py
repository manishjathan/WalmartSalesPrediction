#from Director.LibBasedGCV import libBasedGCV
from Director.LoopBasedGCV import loopBasedGCV
from Director.DataPreparation import prepareTrainTestData
from TrainingUtilities.UtilityFunctions import TimeSeriesUtilities
import pickle
import pandas as pd

if __name__ == '__main__':
    """
    Xtrain, Xtest, ytrain, ytest = prepareTrainTestData(perc_split=0.2)
    dict_params = {'n_estimators' : [i for i in range(30,110,20)],
                   'max_depth' : [i for i in range(5, 15, 2)]}

    loopBasedGCV(dict_params['n_estimators'],
                 dict_params['max_depth'],
                 Xtrain, ytrain)
    
    lpb_df = pickle.load(open('./Data/lpb_hyper_param_df','rb'))
    print(lpb_df.head())

    """
    train_df = pd.read_csv('./Data/train.csv','rb')
    train_df['weights'] = train_df.apply(lambda row : 5 if train_df['IsHoliday'] == True else 1)
    print(train_df.head())
    """
    sd11 = pickle.load(open('./Data/sd11.pkl','rb'))
    sd11_train = sd11[:100]
    sd11_cv = sd11[100:]
    print(sd11_train)
    #TimeSeriesUtilities.rolling_forecast_generic(sd11)
    """
