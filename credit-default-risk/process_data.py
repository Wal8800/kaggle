import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc
import concurrent.futures
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100000)

folder = "data/"
ignore_feat = [
    "SK_ID_CURR"
]

main_feature_names = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_1",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "REGION_RATING_CLIENT_W_CITY",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE",
    "NAME_FAMILY_STATUS",
    "DAYS_ID_PUBLISH",
    "SK_ID_CURR",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "REGION_POPULATION_RELATIVE",
    "NAME_CONTRACT_TYPE",
    "ORGANIZATION_TYPE"
]


_cache_installment = None
_cache_credit_card = None
_cache_pos_cash = None
_cache_pa = None


def read_train():
    return pd.read_csv(folder + "application_train.csv")


def read_test():
    return pd.read_csv(folder + "application_test.csv")


def process_train(features):
    features = features[main_feature_names].copy()

    features["YEARS_BIRTH"] = features["DAYS_BIRTH"] / -365

    features['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    features['CREDIT_INCOME_PERCENT'] = features['AMT_CREDIT'] / features['AMT_INCOME_TOTAL']
    features['ANNUITY_INCOME_PERCENT'] = features['AMT_ANNUITY'] / features['AMT_INCOME_TOTAL']
    features["GOOD_INCOME_PERCENT"] = features["AMT_GOODS_PRICE"] / features["AMT_INCOME_TOTAL"]
    features['DAYS_EMPLOYED_PERCENT'] = features['DAYS_EMPLOYED'] / features['DAYS_BIRTH']
    features['PAYMENT_RATE'] = features['AMT_ANNUITY'] / features['AMT_CREDIT']
    features['CREDIT_TO_GOOD_RATIO'] = features["AMT_CREDIT"] / features["AMT_GOODS_PRICE"]

    features["CREDIT_PER_CHILD"] = features["CNT_CHILDREN"] / features["AMT_CREDIT"]
    features["INCOME_PER_CHILD"] = features["CNT_CHILDREN"] / features["AMT_INCOME_TOTAL"]
    features["PAYMENT_RATE_PER_CHILD"] = features["CNT_CHILDREN"] / features["PAYMENT_RATE"]
    features["CREDIT_PER_FAM"] = features["CNT_FAM_MEMBERS"] / features["AMT_CREDIT"]
    features["INCOME_PER_FAM"] = features["CNT_FAM_MEMBERS"] / features["AMT_INCOME_TOTAL"]
    features["PAYMENT_RATE_PER_FAM"] = features["CNT_FAM_MEMBERS"] / features["PAYMENT_RATE"]
    features["NON_CHILD_MEMBER_DIFF"] = features["CNT_CHILDREN"] - features["CNT_FAM_MEMBERS"]

    features["CREDIT_PER_POP"] = features["AMT_CREDIT"] / features["REGION_POPULATION_RELATIVE"]
    features["INCOME_PER_POP"] = features["AMT_INCOME_TOTAL"] / features["REGION_POPULATION_RELATIVE"]
    features["PAYMENT_RATE_PER_POP"] = features["PAYMENT_RATE"] / features["REGION_POPULATION_RELATIVE"]

    features["EXT_1_2_DIFF"] = features["EXT_SOURCE_1"] - features["EXT_SOURCE_2"]
    features["EXT_1_3_DIFF"] = features["EXT_SOURCE_1"] - features["EXT_SOURCE_3"]
    features["EXT_2_3_DIFF"] = features["EXT_SOURCE_2"] - features["EXT_SOURCE_3"]

    features["EXT_1_2_DIV"] = features["EXT_SOURCE_1"] / features["EXT_SOURCE_2"]
    features["EXT_1_3_DIV"] = features["EXT_SOURCE_1"] / features["EXT_SOURCE_3"]
    features["EXT_2_3_DIV"] = features["EXT_SOURCE_2"] / features["EXT_SOURCE_3"]

    features["EXT_1_2_MUL"] = features["EXT_SOURCE_1"] * features["EXT_SOURCE_2"]
    features["EXT_1_3_MUL"] = features["EXT_SOURCE_1"] * features["EXT_SOURCE_3"]
    features["EXT_2_3_MUL"] = features["EXT_SOURCE_2"] * features["EXT_SOURCE_3"]

    features["EXT_1_REGION_RATING_MUL"] = features["EXT_SOURCE_1"] * features["REGION_RATING_CLIENT_W_CITY"]
    features["EXT_2_REGION_RATING_MUL"] = features["EXT_SOURCE_2"] * features["REGION_RATING_CLIENT_W_CITY"]
    features["EXT_3_REGION_RATING_MUL"] = features["EXT_SOURCE_3"] * features["REGION_RATING_CLIENT_W_CITY"]

    features['NEW_EXT_SOURCES_MEAN'] = features[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

    return features


def post_side_data(features):
    features["PA_PAYMENT_RATE_MED_PERC"] = features["PA_PAST_PAYMENT_RATE_median"] / features["PAYMENT_RATE"]
    features["PA_PAYMENT_RATE_MEAN_PERC"] = features["PA_PAST_PAYMENT_RATE_mean"] / features["PAYMENT_RATE"]

    features["BU_CREDIT_INCOME_PERC"] = features["ACTIVE_AMT_CREDIT_SUM"] / features["AMT_INCOME_TOTAL"]
    features["TOTAL_CREDIT"] = features["ACTIVE_AMT_CREDIT_SUM"] + features['AMT_CREDIT']
    features["TOTAL_CREDIT_INCOME_PERC"] = features["TOTAL_CREDIT"] / features["AMT_INCOME_TOTAL"]
    features["BU_ANNUITY_MEAN_PERC"] = features["BU_CLOSED_ANNUITY_mean"] / features["AMT_ANNUITY"]
    features["BU_ANNUITY_MEDIAN_PERC"] = features["BU_CLOSED_ANNUITY_median"] / features["AMT_ANNUITY"]
    features["BU_ANNUITY_INCOME_PERC_MED"] = features["BU_CLOSED_ANNUITY_median"] / features["AMT_INCOME_TOTAL"]
    features["BU_ANNUITY_INCOME_PERC_MEA"] = features["BU_CLOSED_ANNUITY_mean"] / features["AMT_INCOME_TOTAL"]

    return features


def drop_data(features):
    for feat in ignore_feat:
        if feat in list(features):
            features.drop(feat, inplace=True, axis=1)

    return features


def process_categorical_data(features, drop_null_columns=False, always_label_encode=False, fill_null_columns=False):
    categorical_feat_list = []
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

    return features, categorical_feat_list


def process_data(train_data, test_data, always_label_encode=False, drop_null_columns=False,
                 fill_null_columns=False):

    train_data = process_train(train_data)
    test_data = process_train(test_data)

    train_data, test_data = concurrent_process_and_join_side_data(train_data, test_data)

    train_data = post_side_data(train_data)
    test_data = post_side_data(test_data)

    train_data = drop_data(train_data)
    test_data = drop_data(test_data)

    train_data, categorical_feat_list = process_categorical_data(train_data,
                                                                 drop_null_columns=drop_null_columns,
                                                                 always_label_encode=always_label_encode,
                                                                 fill_null_columns=fill_null_columns)

    test_data, _ = process_categorical_data(test_data,
                                            drop_null_columns=drop_null_columns,
                                            always_label_encode=always_label_encode,
                                            fill_null_columns=fill_null_columns)

    clear_cache()
    return train_data, test_data, categorical_feat_list


def read_installment():
    global _cache_installment
    if _cache_installment is not None:
        return _cache_installment
    else:
        return pd.read_csv(folder + "installments_payments.csv")


def read_cc():
    global _cache_credit_card
    if _cache_credit_card is not None:
        return _cache_credit_card
    else:
        return pd.read_csv(folder + "credit_card_balance.csv")


def read_pos_cash():
    global _cache_pos_cash
    if _cache_pos_cash is not None:
        return _cache_pos_cash
    else:
        return pd.read_csv(folder + "POS_CASH_balance.csv")


def read_pa():
    global _cache_pa
    if _cache_pa is not None:
        return _cache_pa
    else:
        return pd.read_csv(folder + "previous_application.csv")


def read_and_process_installment():
    installments = read_installment()

    installments["INST_DAYS_DIFF"] = installments["DAYS_INSTALMENT"] - installments["DAYS_ENTRY_PAYMENT"]
    installments["INST_PAYMENT_DIFF"] = installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]

    aggs = [
        "mean",
        "min",
        "max",
        "median"
    ]

    column_list = {
        "INST_DAYS_DIFF": aggs,
        "INST_PAYMENT_DIFF": aggs,
        "AMT_PAYMENT": ["mean", "min", "max", "median", "sum"],
        "AMT_INSTALMENT": ["mean", "min", "max", "median", "sum"]
    }

    aggegration = installments.groupby("SK_ID_CURR").agg(column_list)
    aggegration.columns = ["_".join(("INS",) + x) for x in aggegration.columns.ravel()]

    installments = installments.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"])
    for last_n in [1, 10, 20, 30, 50]:
        temp = installments.groupby("SK_ID_CURR").tail(last_n).copy()
        temp = temp.groupby("SK_ID_CURR").agg(column_list)
        temp.columns = ["_".join(("INS",) + x) for x in temp.columns.ravel()]
        temp = temp.add_prefix("LAST_{}_".format(last_n))

        aggegration = aggegration.join(temp, on="SK_ID_CURR", how="left")

    for last_n_days in [30, 60, 90]:
        temp = installments[installments["DAYS_INSTALMENT"] > -last_n_days].groupby("SK_ID_CURR").agg(column_list)
        temp.columns = ["_".join(("INS",) + x) for x in temp.columns.ravel()]
        temp = temp.add_prefix("LAST_{}_DAYS_".format(last_n_days))

        aggegration = aggegration.join(temp, on="SK_ID_CURR", how="left")

    return aggegration


def read_and_process_credit_card():
    credit_card = read_cc()
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

    credit_card = credit_card.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"])
    for last_n in [1, 10, 25, 50]:
        temp = credit_card.groupby("SK_ID_CURR").tail(last_n).copy()

        temp = temp.groupby("SK_ID_CURR")[column_list].agg(aggs)
        temp.columns = ["_".join(x) for x in temp.columns.ravel()]
        temp = temp.add_prefix("LAST_{}_CC_".format(last_n))

        aggegration = aggegration.join(temp, on="SK_ID_CURR", how="left")

    for last_n_days in [30, 60, 90]:
        temp = credit_card[credit_card["MONTHS_BALANCE"] > -last_n_days].groupby("SK_ID_CURR")[column_list].agg(aggs)
        temp.columns = ["_".join(x) for x in temp.columns.ravel()]
        temp = temp.add_prefix("LAST_{}_DAYS_CC_".format(last_n_days))

        aggegration = aggegration.join(temp, on="SK_ID_CURR", how="left")

    return aggegration


def read_and_process_pos_cash():
    pos_cash = read_pos_cash()
    pos_cash["INST_LEFT_DIFF"] = pos_cash["CNT_INSTALMENT"] - pos_cash["CNT_INSTALMENT_FUTURE"]
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
        "INST_LEFT_DIFF"
    ]

    aggegration = pos_cash.groupby("SK_ID_CURR")[column_list].agg(aggs)
    aggegration.columns = ["_".join(("PSCH",) + x) for x in aggegration.columns.ravel()]

    pos_cash = pos_cash.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"])

    temp_list = [
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
        "INST_LEFT_DIFF"
    ]

    for last_n in [1, 10, 50]:
        temp = pos_cash.groupby("SK_ID_CURR").tail(last_n).copy()

        temp = temp.groupby("SK_ID_CURR")[temp_list].agg(aggs)
        temp.columns = ["_".join(x) for x in temp.columns.ravel()]
        temp = temp.add_prefix("LAST_{}_PSCH_".format(last_n))

        aggegration = aggegration.join(temp, on="SK_ID_CURR", how="left")

    for last_n_days in [30, 60, 90]:
        temp = pos_cash[pos_cash["MONTHS_BALANCE"] > -last_n_days].groupby("SK_ID_CURR")[temp_list].agg(aggs)
        temp.columns = ["_".join(x) for x in temp.columns.ravel()]
        temp = temp.add_prefix("LAST_{}_DAYS_PSCH_".format(last_n_days))

        aggegration = aggegration.join(temp, on="SK_ID_CURR", how="left")

    return aggegration


def read_and_process_past_app():
    past_app = read_pa()
    past_app["ACT_CREDIT_PERC"] = past_app["AMT_CREDIT"] / past_app["AMT_APPLICATION"]
    past_app["DOWN_PAY_PERC"] = past_app["AMT_DOWN_PAYMENT"] / past_app["AMT_CREDIT"]
    past_app["PAST_PAYMENT_RATE"] = past_app["AMT_ANNUITY"] / past_app["AMT_CREDIT"]
    past_app["APP_ANNUITY_PERC"] = past_app["AMT_APPLICATION"] / past_app["AMT_ANNUITY"]
    past_app["GOOD_ANNUITY_PERC"] = past_app["AMT_GOODS_PRICE"] / past_app["AMT_ANNUITY"]
    past_app["GOOD_CRED_RATIO"] = past_app["AMT_GOODS_PRICE"] / past_app["AMT_CREDIT"]
    past_app["PAID_CREDIT"] = past_app["AMT_APPLICATION"] / past_app["AMT_CREDIT"]

    aggs = [
        "mean",
        "min",
        "max",
        "median"
    ]

    column_aggs = {
        "AMT_ANNUITY": ["mean", "min", "max", "median", "sum"],
        "AMT_APPLICATION": ["mean", "min", "max", "median", "sum"],
        "AMT_CREDIT": ["mean", "min", "max", "median", "sum"],
        "AMT_DOWN_PAYMENT": aggs,
        "AMT_GOODS_PRICE": ["mean", "min", "max", "median", "sum"],
        "CNT_PAYMENT": aggs,
        "ACT_CREDIT_PERC": aggs,
        "DOWN_PAY_PERC": aggs,
        "DAYS_DECISION": aggs,
        "PAST_PAYMENT_RATE": aggs,
        "APP_ANNUITY_PERC": aggs,
        "GOOD_ANNUITY_PERC": aggs,
        "GOOD_CRED_RATIO": aggs,
        "PAID_CREDIT": aggs
    }

    aggegration = past_app.groupby("SK_ID_CURR").agg(column_aggs)
    aggegration.columns = ["_".join(("PA",) + x) for x in aggegration.columns.ravel()]

    total_prev_app = past_app.groupby("SK_ID_CURR").size()
    total_prev_app = total_prev_app.reset_index(name='PREV_APP_COUNT').set_index("SK_ID_CURR")

    num_of_refused = past_app.loc[past_app["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR").size()
    num_of_refused = num_of_refused.reset_index(name='PREV_APP_REFUSED').set_index("SK_ID_CURR")
    
    total_prev_app = total_prev_app.join(other=num_of_refused, on="SK_ID_CURR", how="left")
    total_prev_app["PREV_APP_REFUSED"].fillna(0, inplace=True)
    
    total_prev_app["PREV_APP_REFUSE_PERC"] = total_prev_app["PREV_APP_REFUSED"] / total_prev_app["PREV_APP_COUNT"]
    aggegration = aggegration.join(other=total_prev_app, on="SK_ID_CURR", how="left")

    past_app = past_app.sort_values(["SK_ID_CURR", "DAYS_DECISION"])
    for last_n in [1, 10]:
        temp = past_app.groupby("SK_ID_CURR").tail(last_n).copy()
        temp = temp.groupby("SK_ID_CURR").agg(column_aggs)
        temp.columns = ["_".join(("PA",) + x) for x in temp.columns.ravel()]
        temp = temp.add_prefix("LAST_{}_".format(last_n))

        aggegration = aggegration.join(temp, on="SK_ID_CURR", how="left")

    return aggegration


def read_and_process_bureau():
    bureau = pd.read_csv(folder + "bureau.csv")
    bureau["DAYS_CREDIT"] = bureau["DAYS_CREDIT"] * -1
    bureau["DAYS_ENDDATE_FACT"] = bureau["DAYS_ENDDATE_FACT"] * -1

    aggs = {
        "DAYS_CREDIT": ["mean", "min", "max", "median"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_ENDDATE_FACT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["mean", "min", "max", "median", "sum"],
        "CNT_CREDIT_PROLONG": ["mean", "min", "max", "median", "sum"],
        "AMT_CREDIT_SUM": ["mean", "min", "max", "median", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["min", "mean", "median", "max", "sum"],
        "AMT_CREDIT_SUM_LIMIT": ["min", "mean", "median", "max", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "min", "max", "median", "sum"]
    }

    aggegration = bureau.groupby("SK_ID_CURR").agg(aggs)
    aggegration.columns = ["_".join(("BU",) + x) for x in aggegration.columns.ravel()]

    # finding number of active credit bureau
    bureau_status = bureau[["SK_ID_CURR", "CREDIT_ACTIVE"]].pivot_table(index=["SK_ID_CURR"], columns="CREDIT_ACTIVE",
                                                                        aggfunc=len, fill_value=0)

    bureau_status = bureau_status.rename_axis(None, axis=1)
    bureau_status.columns = ["CREDIT_ACTIVE_" + x for x in bureau_status.columns]
    bureau_status["BU_CREDIT_TOTAL"] = bureau_status[bureau_status.columns].sum(axis=1)
    bureau_status["CLOSED_PERC"] = bureau_status["CREDIT_ACTIVE_Closed"] / bureau_status["BU_CREDIT_TOTAL"]
    bureau_status["CLOSED_ACTIVE_RATIO"] = bureau_status["CREDIT_ACTIVE_Active"] / bureau_status["CREDIT_ACTIVE_Closed"]

    aggegration = aggegration.join(other=bureau_status, on="SK_ID_CURR", how="left")
    del bureau_status

    # aggregation of monthly balance status
    bureau_balance = pd.read_csv(folder + "bureau_balance.csv")
    bureau_balance_pivot = bureau[["SK_ID_CURR", "SK_ID_BUREAU"]].join(bureau_balance.set_index("SK_ID_BUREAU"),
                                                                       on="SK_ID_BUREAU", how="left")
    bureau_balance_pivot = bureau_balance_pivot[["SK_ID_CURR", "STATUS"]].pivot_table(index=["SK_ID_CURR"],
                                                                                      columns="STATUS", aggfunc=len,
                                                                                      fill_value=0)
    bureau_balance_pivot = bureau_balance_pivot.rename_axis(None, axis=1)
    bureau_balance_pivot.columns = ["BU_STATUS_" + x for x in bureau_balance_pivot.columns]
    aggegration = aggegration.join(other=bureau_balance_pivot, on="SK_ID_CURR", how="left")
    del bureau_balance_pivot

    # finding credit for active bureau
    active_bu_credit_sum = bureau.loc[bureau["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR")["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"].sum()
    active_bu_credit_sum = active_bu_credit_sum.add_prefix("ACTIVE_")
    aggegration = aggegration.join(other=active_bu_credit_sum, on="SK_ID_CURR", how="left")
    del active_bu_credit_sum

    # figuring out the payment rate of the closed bureau credit
    closed_credit = bureau.loc[bureau["CREDIT_ACTIVE"] == "Closed"].copy()
    closed_credit["TIME_TAKEN"] = closed_credit["DAYS_CREDIT"] = closed_credit["DAYS_ENDDATE_FACT"]
    closed_credit["CLOSED_ANNUITY"] = (closed_credit["AMT_CREDIT_SUM"] / closed_credit["TIME_TAKEN"]) * 365

    closed_aggs = {
        "TIME_TAKEN": ["mean", "min", "max", "median"],
        "CLOSED_ANNUITY": ["mean", "min", "max", "median"]
    }

    closed_aggegration = closed_credit.groupby("SK_ID_CURR").agg(closed_aggs)
    closed_aggegration.columns = ["_".join(("BU",) + x) for x in closed_aggegration.columns.ravel()]
    aggegration = aggegration.join(closed_aggegration, on="SK_ID_CURR", how="left")
    del closed_aggegration
    gc.collect()

    bureau = bureau.sort_values(["SK_ID_CURR", "DAYS_CREDIT"])
    for last_n in [1, 10]:
        temp = bureau.groupby("SK_ID_CURR").head(last_n).copy()
        temp = temp.groupby("SK_ID_CURR").agg(aggs)
        temp.columns = ["_".join(("BU",) + x) for x in temp.columns.ravel()]
        temp = temp.add_prefix("LAST_{}_".format(last_n))

        aggegration = aggegration.join(temp, on="SK_ID_CURR", how="left")
        del temp
        gc.collect()

    return aggegration


def calculate_scale_pos_weight(train_data):
    positive_case = train_data[train_data["TARGET"] == 0]
    negative_case = train_data[train_data["TARGET"] == 1]

    print(positive_case.shape)
    print(negative_case.shape)

    return positive_case.shape[0] / negative_case.shape[0]


def process_data_correlation():
    train_data = read_train()
    train_label = train_data["TARGET"]
    train_data.drop("TARGET", axis=1, inplace=True)

    train_data, test_data, cate_feats = process_data(train_data, None, always_label_encode=True)

    train_data["TARGET"] = train_label

    correlations = train_data.corr()["TARGET"].sort_values()

    print('Most Positive Correlations:\n', correlations.tail(15))
    print('\nMost Negative Correlations:\n', correlations.head(15))


def concurrent_process_and_join_side_data(input_train_data, input_test_data):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        installment = executor.submit(read_and_process_installment)
        cc = executor.submit(read_and_process_credit_card)
        pos_cash = executor.submit(read_and_process_pos_cash)
        past_pa = executor.submit(read_and_process_past_app)
        bu = executor.submit(read_and_process_bureau)

        data_to_csv = {
            installment: "installment",
            cc: "cc",
            pos_cash: "pos_cash",
            past_pa: "past_pa",
            bu: "bu"
        }

        for future in concurrent.futures.as_completed(data_to_csv):
            data_name = data_to_csv[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (data_name, exc))
            else:
                input_train_data = input_train_data.join(other=data, on="SK_ID_CURR", how="left")
                input_test_data = input_test_data.join(other=data, on="SK_ID_CURR", how="left")

    return input_train_data, input_test_data


def clear_cache():
    global _cache_pa, _cache_pos_cash, _cache_credit_card, _cache_installment

    _cache_installment = None
    _cache_credit_card = None
    _cache_pos_cash = None
    _cache_pa = None
    gc.collect()


def process_agg_pa_instal_cc_pcsh():
    # pa = read_pa()
    pos_cash = read_pos_cash()
    cc = read_cc()
    installment = read_installment()

    # pa = pa[["SK_ID_PREV ", "SK_ID_CURR", "DAYS_DECISION"]]

    pos_cash = pos_cash.groupby("SK_ID_PREV")

    # LAST N APPLICATION NUMBER OF DAYS DUE

    # AVG NUMBER OF INSTALLMENT LATE PER APPLICATION

    # SUMMING SHIT UP


if __name__ == "__main__":
    installment = read_and_process_pos_cash()
    print(installment.head())
