import lightgbm as lgb
import numpy as np
import process_data
import util
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pprint
import time
import datetime
import warnings

warnings.simplefilter("ignore")


def create_classifier():
    return lgb.LGBMClassifier(
        nthread=3,
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=34,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        silent=-1,
        verbose=-1,
    )


def pred_with_full_data():
    test_clf = create_classifier()
    test_clf.fit(
        train_data,
        train_label,
        categorical_feature=cate_feats
    )

    prediction = test_clf.predict_proba(test_data)
    create_submission(prediction[:, 1], "full")


def create_submission(prediction, extension=""):
    submission = pd.DataFrame()
    submission["SK_ID_CURR"] = test_data_raw["SK_ID_CURR"]
    submission["TARGET"] = prediction

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    submission.to_csv("output/{}_{}.csv".format(current_time, extension), float_format='%.4f', index=None)


train_data = process_data.read_train()
test_data_raw = process_data.read_test()

train_label = train_data["TARGET"]
train_data.drop("TARGET", axis=1, inplace=True)

process_start_time = time.time()
train_data, cate_feats = process_data.process_data(train_data, always_label_encode=True)
test_data, _ = process_data.process_data(test_data_raw, always_label_encode=True)
util.print_time(time.time()-process_start_time)

# pp = pprint.PrettyPrinter(width=200, compact=True)
# pp.pprint(list(train_data))
# scores = cross_val_score(create_classifier(), train_data, train_label, cv=5, scoring='roc_auc')
# print(scores)

print(train_data.isnull().sum())

fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = list(train_data)

skf = StratifiedKFold(n_splits=5, shuffle=True)

validation_scores = []
oof_preds = np.zeros(train_data.shape[0])
sub_preds = np.zeros(test_data_raw.shape[0])

train_start_time = time.time()
for current_fold, (train_index, validation_index) in enumerate(skf.split(train_data, train_label)):
    train_dataset = train_data.iloc[train_index]
    train_dataset_label = train_label.iloc[train_index]

    valid_dataset = train_data.iloc[validation_index]
    valid_dataset_label = train_label.iloc[validation_index]

    clf = create_classifier()

    evaluation_set = [
        (train_dataset, train_dataset_label),
        (valid_dataset, valid_dataset_label)
    ]
    clf.fit(
        train_dataset,
        train_dataset_label,
        eval_set=evaluation_set,
        eval_metric='auc',
        early_stopping_rounds=200,
        verbose=400,
        categorical_feature=cate_feats
    )

    fold_importance_df["fold_{}".format(current_fold)] = clf.feature_importances_

    valid_prediction = clf.predict_proba(valid_dataset)[:, 1]
    oof_preds[validation_index] = clf.predict_proba(valid_dataset, num_iteration=clf.best_iteration_)[:, 1]
    score = roc_auc_score(valid_dataset_label, valid_prediction)
    validation_scores.append(score)
    print("Fold {}, validation score is {}".format(current_fold, score))

    sub_preds += clf.predict_proba(test_data, num_iteration=clf.best_iteration_)[:, 1] / skf.n_splits

util.print_time(time.time()-train_start_time)

print("Average validation score: ", np.average(validation_scores))
print('Full AUC score %.6f' % roc_auc_score(train_label, oof_preds))

fold_importance_df = fold_importance_df.set_index("feature")
fold_importance_df["avg_importance"] = fold_importance_df.mean(axis=1)
fold_importance_df.sort_values("avg_importance", ascending=False, inplace=True)
print(fold_importance_df)

# create_submission(sub_preds)
# pred_with_full_data()
