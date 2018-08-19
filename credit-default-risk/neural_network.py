import tensorflow as tf
from tensorflow import keras
import process_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import util
import warnings

warnings.simplefilter("ignore")


def create_nn_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1024, activation='relu', input_dim=3))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model


train_data = process_data.read_train()
test_data_raw = process_data.read_test()

train_label = train_data["TARGET"]
train_data.drop("TARGET", axis=1, inplace=True)

train_data = train_data[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
train_data = train_data.fillna(-1)

skf = StratifiedKFold(n_splits=3, shuffle=True)

validation_scores = []
oof_preds = np.zeros(train_data.shape[0])

train_start_time = time.time()
for current_fold, (train_index, validation_index) in enumerate(skf.split(train_data, train_label)):
    train_dataset = train_data.iloc[train_index]
    train_dataset_label = train_label.iloc[train_index]

    valid_dataset = train_data.iloc[validation_index]
    valid_dataset_label = train_label.iloc[validation_index]

    clf = create_nn_model()
    clf.fit(train_dataset, train_dataset_label, epochs=10, batch_size=32, verbose=0)

    prediction = clf.predict(valid_dataset)
    oof_preds[validation_index] = prediction[0]

    valid_prediction = clf.predict(valid_dataset)
    score = roc_auc_score(valid_dataset_label, valid_prediction)
    print("Fold {}, validation score is {}".format(current_fold, score))

util.print_time(time.time()-train_start_time)
print(oof_preds)
print('Full AUC score %.6f' % roc_auc_score(train_label, oof_preds))
