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

_cache_installment_result = None
_cache_credit_card_result = None
_cache_pos_cash_result = None
_cache_pa_result = None
_cache_bureau_result = None


def read_train():
    return pd.read_csv(folder + "application_train.csv")


def read_test():
    return pd.read_csv(folder + "application_test.csv")


def process_data(data, application_fields, always_label_encode=False, drop_null_columns=False,
                 fill_null_columns=False):
    features = data[application_fields].copy()
    global _cache_installment_result, _cache_credit_card_result, \
        _cache_pos_cash_result, _cache_pa_result, _cache_bureau_result

    if _cache_installment_result is None:
        _cache_installment_result = read_and_process_installment()

    if _cache_credit_card_result is None:
        _cache_credit_card_result = read_and_process_credit_card()

    if _cache_pos_cash_result is None:
        _cache_pos_cash_result = read_and_process_pos_cash()

    if _cache_pa_result is None:
        _cache_pa_result = read_and_process_past_app()

    if _cache_bureau_result is None:
        _cache_bureau_result = read_and_process_bureau()

    features = features.set_index("SK_ID_CURR")
    features = features.join(other=_cache_installment_result, on="SK_ID_CURR", how="left")
    features = features.join(other=_cache_credit_card_result, on="SK_ID_CURR", how="left")
    features = features.join(other=_cache_pos_cash_result, on="SK_ID_CURR", how="left")
    features = features.join(other=_cache_pa_result, on="SK_ID_CURR", how="left")
    features = features.join(other=_cache_bureau_result, on="SK_ID_CURR", how="left")

    categorical_feat_list = []

    features["YEARS_BIRTH"] = features["DAYS_BIRTH"] / -365

    features['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    features['DAYS_EMPLOYED_ANOM'] = features["DAYS_EMPLOYED"] == 365243

    features['CREDIT_INCOME_PERCENT'] = features['AMT_CREDIT'] / features['AMT_INCOME_TOTAL']
    features['ANNUITY_INCOME_PERCENT'] = features['AMT_ANNUITY'] / features['AMT_INCOME_TOTAL']
    features['CREDIT_TERM'] = features['AMT_ANNUITY'] / features['AMT_CREDIT']
    features['DAYS_EMPLOYED_PERCENT'] = features['DAYS_EMPLOYED'] / features['DAYS_BIRTH']
    features['PAYMENT_RATE'] = features['AMT_ANNUITY'] / features['AMT_CREDIT']

    for feat in ignore_feat:
        if feat in list(features):
            features.drop(feat, inplace=True)

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


def read_and_process_installment():
    installments = pd.read_csv(folder + "installments_payments.csv")

    installments["INST_DAYS_DIFF"] = installments["DAYS_INSTALMENT"] - installments["DAYS_ENTRY_PAYMENT"]
    installments["INST_PAYMENT_DIFF"] = installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]

    aggs = [
        "mean",
        "min",
        "max",
        "median"
    ]

    column_list = ["INST_DAYS_DIFF", "INST_PAYMENT_DIFF"]

    aggegration = installments.groupby("SK_ID_CURR")[column_list].agg(aggs)
    aggegration.columns = ["_".join(("INS",) + x) for x in aggegration.columns.ravel()]

    return aggegration


def read_and_process_credit_card():
    credit_card = pd.read_csv(folder + "credit_card_balance.csv")
    credit_card["CREDIT_USAGE"] = credit_card["AMT_DRAWINGS_CURRENT"] / credit_card["AMT_CREDIT_LIMIT_ACTUAL"]
    credit_card["BALANCE_LIMIT_PERCENT"] = credit_card["AMT_BALANCE"] / credit_card["AMT_CREDIT_LIMIT_ACTUAL"]
    aggs = [
        "mean",
        "min",
        "max",
        "median"
    ]

    # note AMT_DRAW_CURRENT = AMT_DRAWINGS_ATM_CURRENT + AMT_DRAWINGS_OTHER_CURRENT + AMT_DRAWINGS_POS_CURRENT

    column_list = [
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "AMT_TOTAL_RECEIVABLE",
        "CNT_INSTALMENT_MATURE_CUM",
        "CNT_DRAWINGS_CURRENT",
        "SK_DPD",
        "SK_DPD_DEF",
        "CREDIT_USAGE",
        "BALANCE_LIMIT_PERCENT"
    ]

    aggegration = credit_card.groupby("SK_ID_CURR")[column_list].agg(aggs)
    aggegration.columns = ["_".join(("CC",) + x) for x in aggegration.columns.ravel()]

    return aggegration


def read_and_process_pos_cash():
    pos_cash = pd.read_csv(folder + "POS_CASH_balance.csv")
    pos_cash["INST_LEFT_PERC"] = pos_cash["CNT_INSTALMENT"] / pos_cash["CNT_INSTALMENT_FUTURE"]
    aggs = [
        "mean",
        "min",
        "max",
        "median"
    ]

    column_list = [
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
        "INST_LEFT_PERC"
    ]

    aggegration = pos_cash.groupby("SK_ID_CURR")[column_list].agg(aggs)
    aggegration.columns = ["_".join(("PSCH",) + x) for x in aggegration.columns.ravel()]

    return aggegration


def read_and_process_past_app():
    past_app = pd.read_csv(folder + "previous_application.csv")
    past_app["ACT_CREDIT_PERC"] = past_app["AMT_CREDIT"] / past_app["AMT_APPLICATION"]
    past_app["DOWN_PAY_PERC"] = past_app["AMT_DOWN_PAYMENT"] / past_app["AMT_CREDIT"]
    past_app["CRED_INC_PERC"] = past_app["AMT_CREDIT"] / past_app["AMT_ANNUITY"]
    past_app["APP_INC_PERC"] = past_app["AMT_APPLICATION"] / past_app["AMT_ANNUITY"]

    aggs = [
        "mean",
        "min",
        "max",
        "median"
    ]

    column_list = [
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "CNT_PAYMENT",
        "ACT_CREDIT_PERC",
        "DOWN_PAY_PERC",
        "DAYS_DECISION",
        "CRED_INC_PERC",
        "APP_INC_PERC"
    ]

    aggegration = past_app.groupby("SK_ID_CURR")[column_list].agg(aggs)
    aggegration.columns = ["_".join(("PA",) + x) for x in aggegration.columns.ravel()]

    total_prev_app = past_app.groupby("SK_ID_CURR").size()
    total_prev_app = total_prev_app.reset_index(name='PREV_APP_COUNT').set_index("SK_ID_CURR")

    num_of_refused = past_app.loc[past_app["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR").size()
    num_of_refused = num_of_refused.reset_index(name='PREV_APP_REFUSED').set_index("SK_ID_CURR")
    
    total_prev_app = total_prev_app.join(other=num_of_refused, on="SK_ID_CURR", how="left")
    total_prev_app["PREV_APP_REFUSED"].fillna(0, inplace=True)
    
    total_prev_app["PREV_APP_REFUSE_PERC"] = total_prev_app["PREV_APP_REFUSED"] / total_prev_app["PREV_APP_COUNT"]

    # print(total_prev_app.head(40))
    # print(past_app.loc[past_app["SK_ID_CURR"] == 100030])

    return aggegration.join(other=total_prev_app, on="SK_ID_CURR", how="left")


def read_and_process_bureau():
    bureau = pd.read_csv(folder + "bureau.csv")

    aggs = {
        "DAYS_CREDIT": ["mean", "min", "max", "median"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_ENDDATE_FACT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["mean", "min", "max", "median", "sum"],
        "CNT_CREDIT_PROLONG": ["mean", "min", "max", "median", "sum"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean", "min", "max", "median"],
        "AMT_CREDIT_SUM": ["mean", "min", "max", "median", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["min", "mean", "median", "max", "sum"],
        "AMT_CREDIT_SUM_LIMIT": ["min", "mean", "median", "max", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "min", "max", "median", "sum"]
    }

    aggegration = bureau.groupby("SK_ID_CURR").agg(aggs)
    aggegration.columns = ["_".join(("BU",) + x) for x in aggegration.columns.ravel()]

    # finding number of active credit bureau
    num_of_active = bureau.loc[bureau["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR").size()
    num_of_active = num_of_active.reset_index(name='NUM_ACT_BU_CREDIT').set_index("SK_ID_CURR")

    num_of_closed = bureau.loc[bureau["CREDIT_ACTIVE"] == "Closed"].groupby("SK_ID_CURR").size()
    num_of_closed = num_of_closed.reset_index(name='NUM_CL_BU_CREDIT').set_index("SK_ID_CURR")

    aggegration = aggegration.join(other=num_of_active, on="SK_ID_CURR", how="left")
    aggegration["NUM_ACT_BU_CREDIT"].fillna(0, inplace=True)

    aggegration = aggegration.join(other=num_of_closed, on="SK_ID_CURR", how="left")
    aggegration["NUM_CL_BU_CREDIT"].fillna(0, inplace=True)

    bureau_balance = pd.read_csv(folder + "bureau_balance.csv")

    bureau_balance_pivot = bureau[["SK_ID_CURR", "SK_ID_BUREAU"]].join(bureau_balance.set_index("SK_ID_BUREAU"), on="SK_ID_BUREAU", how="left")
    bureau_balance_pivot = bureau_balance_pivot[["SK_ID_CURR", "STATUS"]].pivot_table(index=["SK_ID_CURR"],
                                                                                      columns="STATUS", aggfunc=len,
                                                                                      fill_value=0)
    bureau_balance_pivot = bureau_balance_pivot.rename_axis(None, axis=1)
    bureau_balance_pivot.columns = ["BU_STATUS_" + x for x in bureau_balance_pivot.columns]

    aggegration = aggegration.join(other=bureau_balance_pivot, on="SK_ID_CURR", how="left")

    return aggegration


if __name__ == "__main__":
    read_and_process_bureau()
