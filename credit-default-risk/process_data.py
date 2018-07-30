import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

folder = "data/"
ignore_feat = [
    "SK_ID_CURR"
]


def read_train():
    return pd.read_csv(folder + "application_train.csv")


def read_test():
    return pd.read_csv(folder + "application_test.csv")


def process_data(data, application_fields, always_label_encode=False, drop_null_columns=False,
                 fill_null_columns=False):
    features = data[application_fields].copy()
    categorical_feat_list = []

    for feat in ignore_feat:
        if feat in list(features):
            features.drop(feat, inplace=True)

    features["YEARS_BIRTH"] = features["DAYS_BIRTH"] / -365

    features['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    features['DAYS_EMPLOYED_ANOM'] = features["DAYS_EMPLOYED"] == 365243

    features['CREDIT_INCOME_PERCENT'] = features['AMT_CREDIT'] / features['AMT_INCOME_TOTAL']
    features['ANNUITY_INCOME_PERCENT'] = features['AMT_ANNUITY'] / features['AMT_INCOME_TOTAL']
    features['CREDIT_TERM'] = features['AMT_ANNUITY'] / features['AMT_CREDIT']
    features['DAYS_EMPLOYED_PERCENT'] = features['DAYS_EMPLOYED'] / features['DAYS_BIRTH']
    features['PAYMENT_RATE'] = features['AMT_ANNUITY'] / features['AMT_CREDIT']

    # categorical data into numerical
    for column_name in features:
        if features[column_name].isnull().any() and drop_null_columns:
            features.drop(column_name, axis=1, inplace=True)
            continue

        dtype = features[column_name].dtype.name
        if is_numeric_dtype(dtype):
            if fill_null_columns:
                features[column_name].fillna(-1, inplace=True)
            continue

        categorical_feat_list.append(column_name)
        if len(features[column_name].unique()) <= 2 or always_label_encode:
            # need to fill nan so it doesn't throw exception
            features[column_name].fillna('Unknown', inplace=True)
            features[column_name] = LabelEncoder().fit_transform(features[column_name])
        else:
            dummy = pd.get_dummies(features[column_name])
            features = pd.concat([features, dummy], axis=1)
            features.drop(column_name, axis=1, inplace=True)

    print("Number of features: {}".format(len(list(features))))
    return features, categorical_feat_list
