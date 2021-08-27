from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class InputModel():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def describe_data(self):
        return(f"Shape of train_data : {self.Xtrain.shape} \nShape of target data: {self.ytrain.shape}")

    def randomSplit(self,test_size):
        try:
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
            self.Xtrain = pd.DataFrame(self.Xtrain, columns = self.X.columns)
            self.Xtest = pd.DataFrame(self.Xtest, columns=self.X.columns)
            return self.Xtrain, self.Xtest, self.ytrain, self.ytest
        except Exception as e:
            print(e)
            raise Exception("Error in random splitting of train and test data")