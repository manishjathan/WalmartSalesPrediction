from abc import abstractmethod,abstractproperty

class MLModel():

    @abstractmethod
    def train_model(self, params):
        pass

    @abstractmethod
    def train_model_gcv(self, params, scoring=None):
        pass

    @abstractmethod
    def train_model_rcv(self, params, scoring=None):
        pass

    @abstractmethod
    def evaluate_model(self,truth,pred):
        pass
    @abstractmethod
    def predict_results(self):
        pass
