import lightgbm as lgb
import numpy as np
import process_data
import time
import util
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import itertools
import multiprocessing
import pandas as pd
import warnings

warnings.simplefilter("ignore")


def train_and_fit(classfier):
    oof_preds = np.zeros(train_data.shape[0])
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    train_start_time = time.time()
    for current_fold, (train_index, validation_index) in enumerate(skf.split(train_data, train_label)):
        train_dataset = train_data.iloc[train_index]
        train_dataset_label = train_label.iloc[train_index]

        valid_dataset = train_data.iloc[validation_index]
        valid_dataset_label = train_label.iloc[validation_index]

        evaluation_set = [
            (train_dataset, train_dataset_label),
            (valid_dataset, valid_dataset_label)
        ]
        classfier.fit(
            train_dataset,
            train_dataset_label,
            eval_set=evaluation_set,
            eval_metric='auc',
            early_stopping_rounds=200,
            verbose=False,
            categorical_feature=cate_feats
        )

        oof_preds[validation_index] = clf.predict_proba(valid_dataset, num_iteration=clf.best_iteration_)[:, 1]

    return roc_auc_score(train_label, oof_preds), (time.time() - train_start_time)


train_data = process_data.read_train()
test_data_raw = process_data.read_test()

train_label = train_data["TARGET"]
train_data.drop("TARGET", axis=1, inplace=True)

process_start_time = time.time()
train_data, cate_feats = process_data.process_data(train_data, always_label_encode=True)
test_data, _ = process_data.process_data(test_data_raw, always_label_encode=True)
util.print_time(time.time()-process_start_time)

params = {
    "nthread": [multiprocessing.cpu_count()],
    "n_estimator": [5000],
    "num_leaves": range(30, 60, 10),
    "learning_rate": [0.75],
    # "learning_rate": np.arange(0.005, 0.02, 0.005),
    "colsample_bytree": np.arange(0.8, 1.5, 0.05),
    "subsample": np.arange(0.825, 1.025, 0.025),
    "max_depth": range(7, 10),
    "reg_alpha": np.arange(0, 0.10, 0.02),
    "reg_lambda": np.arange(0, 0.10, 0.02),
    "min_split_gain": np.arange(0, 0.08, 0.02),
    "min_child_weight": range(20, 50, 10)
}

keys, values = zip(*params.items())
count = 0
tuning_results = pd.DataFrame(columns=["Parameters", "Scores"])
for v in itertools.product(*values):
    experiment = dict(zip(keys, v))
    count += 1

    clf = lgb.LGBMClassifier(
        nthread=experiment["nthread"],
        n_estimators=experiment["n_estimator"],
        learning_rate=experiment["learning_rate"],
        num_leaves=experiment["num_leaves"],
        colsample_bytree=experiment["colsample_bytree"],
        subsample=experiment["subsample"],
        max_depth=experiment["max_depth"],
        reg_alpha=experiment["reg_alpha"],
        reg_lambda=experiment["reg_lambda"],
        min_split_gain=experiment["min_split_gain"],
        min_child_weight=experiment["min_child_weight"],
        silent=False,
        verbose=-1,
    )

    scores, time_taken = train_and_fit(clf)

    print("Run: {}, Parameters: {}, Score: {}, Time Taken: {}".format(count, experiment, scores, time_taken))
    tuning_results = tuning_results.append({
        "Parameters": experiment,
        "Scores": scores
    }, ignore_index=True)

tuning_results.sort_values("Scores", ascending=False, inplace=True)
print(tuning_results.head(10))
