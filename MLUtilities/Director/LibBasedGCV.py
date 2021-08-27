from TransformationUtilities import Transformation
from TrainingUtilities.FactoryModels import InputDataModel
from EvaluationUtilities.ModelEvaluation import calcWMAE
from TrainingUtilities.FactoryModels import ModelCreatorFactory
from sklearn.metrics import make_scorer
import numpy as np
import pickle

def libBasedGCV(params,Xtrain,Xtest,ytrain,ytest):
    ## Creating a Random Forest Regressor
    scoring_func = make_scorer(calcWMAE, greater_is_better=False)
    regressorCreator = ModelCreatorFactory.RegressorCreator()
    gcvRfRegressor = regressorCreator.create_training_model(model_name="RFR")
    gcvRfRegressor.init(Xtrain, Xtest, ytrain, ytest)

    print("*" * 25, "Started training the model", "*" * 25)
    gcvRfRegressor.train_model_gcv(params,
                                   scoring=scoring_func)
    print("*" * 25, "Finished training the model", "*" * 25)
    gcvRfRegressor.predict_results()
    gcvRfRegressor.evaluate_model()
    gcvRfRegressor.desc_model()
