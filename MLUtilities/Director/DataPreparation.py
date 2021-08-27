from TrainingUtilities.FactoryModels import InputDataModel
from TransformationUtilities import Transformation
import pickle
import numpy as np

def prepareTrainTestData(perc_split=0.2):
        sampleFile = open('./Data/merge.csv', 'rb')
        merge_df = pickle.load(sampleFile)
        sampleFile.close()

        X = merge_df.drop(['Weekly_Sales'], axis=1)
        y = merge_df[['Weekly_Sales']]
        inpModel = InputDataModel.InputModel(X, y)

        ############################Random Splitting of data##################################
        Xtrain, Xtest, ytrain, ytest = inpModel.randomSplit(perc_split)

        ytrain = np.ravel(ytrain)
        train_trans_df = Transformation.createNumericalAggr(Xtrain[['Store', 'Dept', 'IsHoliday',
                                                                    'Temperature', 'Fuel_Price',
                                                                    'CPI', 'Unemployment', 'Size']],
                                                            ['Store', 'Dept'])

        # Imputing null values with mean as only a small part of it contains empty values
        train_trans_df.fillna(train_trans_df.mean(), inplace=True)
        Xtrain = Xtrain.merge(train_trans_df, how='left', on=['Store', 'Dept'])
        Xtrain = Transformation.createDateFeatures(Xtrain, 'Date').drop('Date', axis=1)

        # Since they contain a lot of null values
        markdown_cols = ['MarkDown' + str(i) for i in range(1, 6)]

        ############################Transformation of test data############################
        ytest = np.ravel(ytest)
        test_trans_df = Transformation.createNumericalAggr(Xtest[['Store', 'Dept', 'IsHoliday',
                                                                  'Temperature', 'Fuel_Price',
                                                                  'CPI', 'Unemployment', 'Size']],
                                                           ['Store', 'Dept'])

        # Imputing null values with mean as only a small part of it contains empty values
        test_trans_df.fillna(test_trans_df.mean(), inplace=True)
        Xtest = Xtest.merge(test_trans_df, how='left', on=['Store', 'Dept'])
        Xtest = Transformation.createDateFeatures(Xtest, 'Date').drop('Date', axis=1)

        ############################Dropping columns with more than 50% null values############
        Xtrain.drop(markdown_cols, axis=1, inplace=True)
        Xtest.drop(markdown_cols, axis=1, inplace=True)

        print("Shape of Train data : ", Xtrain.shape)
        print("Shape of Test data : ", Xtest.shape)
        print("*" * 25, "Created train and test data", "*" * 25)

        return Xtrain, Xtest, ytrain, ytest