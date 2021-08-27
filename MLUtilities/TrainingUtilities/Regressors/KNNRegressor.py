from TrainingUtilities.FactoryModels.ModelInterface import MLModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class KNNRegressor(MLModel):

    def __init__(self, Xtrain, Xtest, ytrain, ytest):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest

    def train_model(self, params):
        """
        :param params: parameters to try for model in the form of dicitonary
        :return:
        """
        self.knnRegressor = KNeighborsRegressor(**params)
        self.knnRegressor.fit(self.Xtrain, self.ytrain)

    def train_model_gcv(self, params, scoring=None):
        self.knnRegressor = GridSearchCV(KNeighborsRegressor(),
                                         params,
                                         scoring = scoring,
                                         refit=True,
                                         n_jobs=-1)
        self.knnRegressor.fit(self.Xtrain, self.ytrain)

    def train_model_rcv(self, params, scoring=None):
        self.knnRegressor = RandomizedSearchCV(KNeighborsRegressor(),
                                               params,
                                               scoring = scoring,
                                               refit=True,
                                               n_jobs=-1)
        self.knnRegressor.fit(self.Xtrain, self.ytrain)

    def predict_results(self):
        self.ypred = self.knnRegressor.predict(self.Xtest)

    def evaluate_model(self) -> str:
        self.model_score = np.sqrt(calcRMSE(self.ytest, self.ypred))
