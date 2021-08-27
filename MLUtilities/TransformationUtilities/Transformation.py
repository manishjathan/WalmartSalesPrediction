from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import numpy as np
import warnings
warnings.filterwarnings("ignore")

NUM_DTYPES_LIST = ['float', 'int','float64','int64']

def standardizeData(train_data, test_data):
    """

    :param train_data:
    :param test_data:
    :return: Standardized train and test dataframe
    """
    try:
        train_data = train_data.select_dtypes(include = NUM_DTYPES_LIST)
        test_data = test_data.select_dtypes(include = NUM_DTYPES_LIST)

        standard_scalar = StandardScaler()
        std_train_data = standard_scalar.fit_transform(train_data)
        std_test_data = standard_scalar.transform(test_data)
        std_train_df = pd.DataFrame(std_train_data, columns=train_data.columns)
        std_test_df = pd.DataFrame(std_test_data, columns=test_data.columns)

        return std_train_df, std_test_df
    except Exception as e:
        print(e)
        raise Exception("Exception in Standardizing the data")

def normalizeData(train_data, test_data):
    try:
        train_data = train_data.select_dtypes(include=NUM_DTYPES_LIST)
        test_data = test_data.select_dtypes(include=NUM_DTYPES_LIST)

        normalizer = Normalizer()
        norm_train_data = normalizer.fit_transform(train_data)
        norm_test_data = normalizer.fit_transform(test_data)
        norm_train_df = pd.DataFrame(norm_train_data, columns = train_data.columns)
        norm_test_df = pd.DataFrame(norm_test_data, columns = test_data.columns)
        return norm_train_df, norm_test_df
    except Exception as e:
        print(e)
        raise Exception("Exception in Normalizing the data")
def createNumericalAggr(df, groupby_cat,aggr_funcs = ['min', 'max', 'median', 'sum', 'std', 'var','skew']):
    """
    Input :
            df          : DataFrame to be used for operation
            groupby_cat : Column to be used for groupby
            num_feat    : Feature across which aggregation will be performed
    returns aggregated features with min,max,median,sum and standard-deviation
    """
    try:
        num_feat = df.select_dtypes(NUM_DTYPES_LIST).columns
        aggr_dict = dict([(feat, aggr_funcs) for feat in num_feat])
        aggr_df = df.groupby(groupby_cat).agg(aggr_dict).reset_index()
        aggr_df.columns = ['_'.join(col).strip('_') for col in aggr_df.columns.values]
        return aggr_df
    except Exception as e:
        print(e)
        raise Exception("Exception in Numerical Aggregation of columns")
def createDateFeatures(df,date_col):
    """

    :param df:
    :param date_col:
    :return: returns date features like month,day,year and week
    """
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['year'] = df[date_col].dt.year
        df['week'] = df[date_col].dt.week
        return df
    except Exception as e:
        print(e)
        raise Exception("Exception in creating date time features")
def createOneHotEncodedFeatures(df, feature):
    """

    :param train_df:
    :param test_df:
    :param feature:
    :return: OneHotEncoded features
    """
    try:
        ohe_df = pd.get_dummies(df[feature])
        return ohe_df
    except Exception as e:
        print(e)
        raise Exception("Exception in creating oneHotEncoded features")
def createPCAfeatures(train_df, test_df, features, n_components = 2):
   try:
        pca = PCA(n_components)
        train_pca = pca.fit_transform(train_df[features])
        test_pca = pca.transform(test_df[features])
        train_pca_df = pd.DataFrame(train_pca, columns = ['pca_' + str(i) for i in range(1, n_components+1)])
        test_pca_df = pd.DataFrame(test_pca, columns = ['pca_' + str(i) for i in range(1, n_components+1)])
        return train_pca_df, test_pca_df
   except Exception as e:
        print(e)
        raise("Exception in creating PCA features")
def createSVDfeatures(train_df, test_df, features, n_components = 2,verbose=False):
    try:
        svd = TruncatedSVD(n_components = n_components, random_state=42)
        svd_train_data = svd.fit_transform(train_df[features])
        svd_test_data = svd.transform(test_df[features])

        svd_train_df = pd.DataFrame(svd_train_data)
        svd_test_df = pd.DataFrame(svd_test_data)
        svd_train_df.columns = ['svd_' + str(col) for col in svd_train_df.columns]
        svd_test_df.columns = ['svd_' + str(col) for col in svd_test_df.columns]

        if verbose == True:
            print("Explained Variance ration with ", n_components, " components : ", np.sum(svd.explained_variance_ratio_) * 100)
        return svd_train_df, svd_test_df, svd

    except Exception as e:
        print(e)
        raise Exception("Exception in creating SVD features")


