import tensorflow as tf
from tensorflow import keras
import process_data
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

train_data = process_data.read_train()
test_data_raw = process_data.read_test()

train_label = train_data["TARGET"]
train_data.drop("TARGET", axis=1, inplace=True)

train_data = train_data[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]

print(train_data.shape)

fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = list(train_data)

skf = StratifiedKFold(n_splits=5, shuffle=True)

validation_scores = []
oof_preds = np.zeros(train_data.shape[0])
sub_preds = np.zeros(test_data_raw.shape[0])


model = keras.Sequential()
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss="cate")