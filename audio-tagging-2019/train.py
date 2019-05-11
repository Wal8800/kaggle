import pandas as pd
import numpy as np
import tensorflow as tf
import time

from model import simple_2d_conv, keras_cnn
from data_loader import MelDataGenerator, load_melspectrogram_files
from sklearn.model_selection import KFold
from natural import date
from sklearn.model_selection import train_test_split
from train_util import calculate_overall_lwlrap_sklearn, EarlyStoppingByLWLRAP


def kfold_validation(input_data, input_labels):
    kf = KFold(n_splits=5, shuffle=True)
    fold_scores = []
    current_fold = 1
    start_time = time.time()
    for train_index, test_index in kf.split(input_data):
        tf.keras.backend.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))

        x_train, x_test = input_data[train_index], input_data[test_index]
        y_train, y_test = input_labels.values[train_index], input_labels.values[test_index]

        data_dir = "processed/melspectrogram/"

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

        train_generator = MelDataGenerator(x_train, y_train, data_dir, batch_size=32)
        x, y = train_generator[0]
        max_row = x[0].shape[0]
        max_column = x[0].shape[1]
        max_depth = x[0].shape[2]
        num_classes = y[0].shape[0]

        # x_train = load_melspectrogram_files(data_dir, x_train)
        # max_row = x_train[0].shape[0]
        # max_column = x_train[0].shape[1]
        # max_depth = x_train[0].shape[2]
        # num_classes = y_train[0].shape[0]
        print("Traing shape: ", (max_row, max_column, max_depth))

        # val_generator = MelDataGenerator(x_val, y_val, data_dir)
        x_val = load_melspectrogram_files(data_dir, x_val)

        # create 2d conv model
        model = keras_cnn((max_row, max_column, max_depth), num_classes)
        opt = tf.keras.optimizers.Adam()
        model.compile(loss='binary_crossentropy', optimizer=opt)

        callbacks = [
            # tf.keras.callbacks.TensorBoard(log_dir='./logs/fold_{}'.format(current_fold), histogram_freq=0,
            #                                batch_size=32, write_graph=True, write_grads=False, write_images=False,
            #                                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
            #                                embeddings_data=None, update_freq='epoch'),
            EarlyStoppingByLWLRAP(validation_data=(x_val, y_val), patience=20)
            # tf.keras.callbacks.ModelCheckpoint('./models/best_{}.h5'.format(current_fold),
            #                                    monitor='val_loss', verbose=1, save_best_only=True)
        ]

        model.fit_generator(train_generator, epochs=200, callbacks=callbacks)
        # model.fit(x_train, y_train, batch_size=32, epochs=200, callbacks=callbacks, validation_data=(x_val, y_val))

        test_generator = MelDataGenerator(x_test, y_test, "processed/melspectrogram/")
        y_pred = model.predict_generator(test_generator)
        # x_test = load_melspectrogram_files(data_dir, x_test)
        # y_pred = model.predict(x_test)

        lwlrap = calculate_overall_lwlrap_sklearn(y_test, y_pred)
        print("Fold {} Score: {}".format(current_fold, lwlrap))
        current_fold += 1
        fold_scores.append(lwlrap)
    print(fold_scores)
    print("Average Fold Score:", np.mean(fold_scores))
    print("Time taken: {}".format(date.compress(time.time() - start_time)))


def train_all_data_set(input_data, input_labels, input_shape, num_classes):
    # need to set this configuration to run tensorflow RTX 2070 GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    start_time = time.time()
    # create 2d conv model
    model = simple_2d_conv(input_shape, num_classes)

    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs/all_data', histogram_freq=0,
                                       batch_size=32, write_graph=True, write_grads=False, write_images=False,
                                       embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                                       embeddings_data=None, update_freq='epoch'),
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10),
        tf.keras.callbacks.ModelCheckpoint('./models/model.h5',
                                           monitor='val_loss', verbose=1, save_best_only=True),
    ]

    model.fit(input_data, input_labels, batch_size=32, epochs=200, callbacks=callbacks, validation_split=0.2)
    print("Time taken: {}".format(date.compress(time.time() - start_time)))


def train():
    train_curated = pd.read_csv("data/train_curated.csv")
    labels = train_curated['labels'].str.get_dummies(sep=',')

    kfold_validation(train_curated['fname'], labels)

    # train_all_data_set(melspec_ndarray, labels, (max_row, max_col, max_depth), len(labels.columns))


if __name__ == "__main__":
    train()
