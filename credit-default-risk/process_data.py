import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

folder = "data/"


def read_train():
    return pd.read_csv(folder + "application_train.csv")


def read_test():
    return pd.read_csv(folder + "application_test.csv")


def process_data(data, application_fields, always_label_encode=False):
    features = data[application_fields].copy()

    # categorical data into numerical
    for column_name in features:
        # if features[column_name].isnull().any():
        #     features.drop(column_name, axis=1, inplace=True)
        #     continue

        dtype = features[column_name].dtype.name
        if is_numeric_dtype(dtype):
            continue

        if len(features[column_name].unique()) <= 2 or always_label_encode:
            features[column_name] = LabelEncoder().fit_transform(features[column_name])
        else:
            dummy = pd.get_dummies(features[column_name])
            features = pd.concat([features, dummy], axis=1)
            features.drop(column_name, axis=1, inplace=True)

    return features


train = read_train()
process_data(train, ["NAME_CONTRACT_TYPE", "NAME_HOUSING_TYPE", "COMMONAREA_MEDI"])
