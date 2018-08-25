import tensorflow as tf
from tensorflow import keras
import process_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, Callback
import pandas as pd
import numpy as np
import util
import warnings

warnings.simplefilter("ignore")


class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def create_nn_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(512, activation='relu', input_dim=3))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy')

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

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    roc_callback = RocCallback(training_data=(train_dataset, train_dataset_label),
                               validation_data=(valid_dataset, valid_dataset_label))

    clf = create_nn_model()
    clf.fit(train_dataset, train_dataset_label, epochs=10, batch_size=32,
            validation_data=(valid_dataset, valid_dataset_label), callbacks=[early_stopping, roc_callback])

    prediction = clf.predict(valid_dataset)
    oof_preds[validation_index] = prediction[0]

    valid_prediction = clf.predict(valid_dataset)
    score = roc_auc_score(valid_dataset_label, valid_prediction)
    print("Fold {}, validation score is {}".format(current_fold, score))

util.print_time(time.time()-train_start_time)
print(oof_preds)
print('Full AUC score %.6f' % roc_auc_score(train_label, oof_preds))
