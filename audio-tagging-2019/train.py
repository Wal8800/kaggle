import pandas as pd
import numpy as np
import tensorflow as tf
import time
from os import listdir
from os.path import isfile, join

from model import simple_2d_conv, keras_cnn, create_model_simplecnn
from data_loader import MelDataGenerator, load_melspectrogram_files
from sklearn.model_selection import KFold
from natural import date
from sklearn.model_selection import train_test_split
from train_util import calculate_overall_lwlrap_sklearn, EarlyStoppingByLWLRAP, bce_with_logits, tf_lwlrap


class TrainingConfiguration:
    def __init__(self, generator, load_files, num_epoch=200, training_data_dir="processed/melspectrogram/"):
        self.generator = generator
        self.load_files = load_files
        self.num_epoch = num_epoch
        self.training_data_dir = training_data_dir


def kfold_validation(train_config: TrainingConfiguration, input_data, input_labels):
    kf = KFold(n_splits=3, shuffle=True)
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

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)

        train_generator = train_config.generator(x_train, y_train, batch_size=32, directory=train_config.training_data_dir)
        x, y = train_generator[0]
        max_row = x[0].shape[0]
        max_column = x[0].shape[1]
        max_depth = x[0].shape[2]
        num_classes = y[0].shape[0]

        print("Training shape: ", (max_row, max_column, max_depth))

        x_val = train_config.load_files(x_val, directory=train_config.training_data_dir)

        # create 2d conv model
        model = create_model_simplecnn((max_row, max_column, max_depth), num_classes)
        opt = tf.keras.optimizers.Adam()
        model.compile(loss=bce_with_logits, optimizer=opt, metrics=[tf_lwlrap])

        callbacks = [
            # tf.keras.callbacks.TensorBoard(log_dir='./logs/fold_{}'.format(current_fold), histogram_freq=0,
            #                                batch_size=32, write_graph=True, write_grads=False, write_images=False,
            #                                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
            #                                embeddings_data=None, update_freq='epoch'),
            EarlyStoppingByLWLRAP(validation_data=(x_val, y_val), patience=5),
            tf.keras.callbacks.ModelCheckpoint('./models/best_{}.h5'.format(current_fold),
                                               monitor='val_tf_lwlrap', verbose=1, save_best_only=True, mode='max')
        ]

        model.fit_generator(train_generator,
                            epochs=train_config.num_epoch,
                            callbacks=callbacks,
                            validation_data=(x_val, y_val))

        test_generator = train_config.generator(x_test, y_test, directory=train_config.training_data_dir)
        y_pred = model.predict_generator(test_generator)

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


def extract_original_file_name(augmented_fname):
    return ".".join(augmented_fname.split(".", 2)[:2])


def train():
    train_curated = pd.read_csv("data/train_curated.csv")
    # file_names = train_curated['fname']
    # labels = train_curated['labels'].str.get_dummies(sep=',')
    augmented_dir = "processed/augmented_melspectrogram/"
    file_names = np.array([f for f in listdir(augmented_dir) if isfile(join(augmented_dir, f))])
    augmented_labels = [train_curated[train_curated['fname'] == extract_original_file_name(file_name)]["labels"].values[0]
                        for file_name in file_names]
    temp = pd.DataFrame(
        {
            'labels': augmented_labels
        }
    )
    labels = temp['labels'].str.get_dummies(sep=',')

    train_config = TrainingConfiguration(
        MelDataGenerator,
        load_melspectrogram_files,
        training_data_dir="processed/augmented_melspectrogram/")

    kfold_validation(train_config, file_names, labels)


if __name__ == "__main__":
    train()
