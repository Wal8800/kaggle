import lightgbm as lgb
import numpy as np
import process_data
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def create_classifier():
    return lgb.LGBMClassifier(
        nthread=4,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        silent=-1,
        verbose=-1)

train_data = process_data.read_train()
train_label = train_data["TARGET"]
train_data.drop("TARGET", axis=1, inplace=True)

feature_names = list(train_data)
feature_names = np.random.choice(feature_names, size=10, replace=False)
feature_names = ["EXT_SOURCE_2", "EXT_SOURCE_1", "EXT_SOURCE_3"]

train_data = process_data.process_data(train_data, feature_names, always_label_encode=True)

print(train_data.head())

skf = StratifiedKFold(n_splits=5, shuffle=True)

validation_scores = []
current_fold = 1
for train_index, validation_index in skf.split(train_data, train_label):
    train_dataset = train_data.iloc[train_index]
    train_dataset_label = train_label.iloc[train_index]

    valid_dataset = train_data.iloc[validation_index]
    valid_dataset_label = train_label.iloc[validation_index]

    clf = create_classifier()

    evaluation_set = [
        (train_dataset, train_dataset_label),
        (valid_dataset, valid_dataset_label)
    ]
    clf.fit(train_data, train_label, eval_set=evaluation_set, eval_metric='auc', early_stopping_rounds=200, verbose=200)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = list(train_data)
    fold_importance_df["importance"] = clf.feature_importances_

    valid_prediction = clf.predict_proba(valid_dataset)[:, 1]

    score = roc_auc_score(valid_dataset_label, valid_prediction)
    validation_scores.append(score)
    print("Fold {}, validation score is {}".format(current_fold, score))
    current_fold += 1


print("Average validation score: ", np.average(validation_scores))
# test_data_raw = process_data.read_test()
# test_clf = create_classifier()
#
# test_data = process_data.process_data(test_data_raw, feature_names, always_label_encode=True)
# test_clf.fit(train_data, train_label)
# prediction = test_clf.predict_proba(test_data)
#
# submission = pd.DataFrame()
# submission["SK_ID_CURR"] = test_data_raw["SK_ID_CURR"]
# submission["TARGET"] = prediction[:, 1]
#
# submission.to_csv("result.csv", float_format='%.4f', index=None)
