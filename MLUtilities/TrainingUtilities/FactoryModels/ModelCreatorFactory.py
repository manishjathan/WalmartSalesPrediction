from __future__ import annotations
from abc import ABC, abstractmethod
from TrainingUtilities.FactoryModels.ModelInterface import  MLModel
from TrainingUtilities.Regressors.RFRegressor import RFRegressor
from TrainingUtilities.Regressors.KNNRegressor import KNNRegressor


import warnings
warnings.filterwarnings("ignore")

class Creator(ABC):
    @abstractmethod
    def create_training_model(self):
        pass

class RegressorCreator():

    def create_training_model(self, model_name) -> MLModel:
        if model_name == 'KNNR':
            return KNNRegressor()
        elif model_name == 'RFR':
            return RFRegressor()
        else:
            raise Exception("Error : Model not found")



