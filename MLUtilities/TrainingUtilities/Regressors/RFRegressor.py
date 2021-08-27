from TrainingUtilities.FactoryModels.ModelInterface import MLModel
from EvaluationUtilities.ModelEvaluation import calcWMAE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold


class RFRegressor(MLModel):

    def init(self, Xtrain, Xtest, ytrain, ytest):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        

    def train_model(self, params):
        """
        :param params: parameters to try for model in the form of dicitonary
        :return:
        """
        self.rfRegressor = RandomForestRegressor(**params)
        self.rfRegressor.fit(self.Xtrain, self.ytrain)

    def train_model_gcv(self, params, scoring=None):
        self.rfRegressor = GridSearchCV(RandomForestRegressor(),
                                        params,
                                        refit=True,
                                        scoring = scoring,
                                        verbose = 3,
                                        cv = KFold(n_splits=3))
        self.rfRegressor.fit(self.Xtrain, self.ytrain)


    def train_model_rcv(self,params,scoring=None):
        self.rfRegressor = RandomizedSearchCV(RandomForestRegressor(),
                                              params,
                                              refit=True,
                                              scoring = scoring,
                                              verbose=3)
        self.rfRegressor.fit(self.Xtrain, self.ytrain)


    def predict_results(self):
        self.ypred = self.rfRegressor.predict(self.Xtest)
        return self.ypred

    def evaluate_model(self):
        """
        self.model_score = np.sqrt(calcRMSE(self.ytest, self.ypred))
        """
        self.model_score = calcWMAE(self.ytest, self.ypred)
        return self.model_score


    def desc_model(self):
        print("*" * 25, "Best Params", "*" * 25)
        print(self.rfRegressor.best_params_)
        print("*" * 25, "Best Score", "*" * 25)
        print(self.rfRegressor.best_score_)
        print("*" * 25, "Important Features", "*" * 25)
        print(self.rfRegressor.best_estimator_.feature_importances_)
