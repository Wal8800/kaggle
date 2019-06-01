import pandas as pd
import numpy as np
import tensorflow as tf
import time

from model import simple_2d_conv, keras_cnn, create_model_simplecnn, resnet_v1
from data_loader import MelDataGenerator, load_melspectrogram_files
from sklearn.model_selection import KFold
from natural import date
from sklearn.model_selection import train_test_split
from train_util import calculate_overall_lwlrap_sklearn, EarlyStoppingByLWLRAP, bce_with_logits, tf_lwlrap, lr_schedule
from train_util import calculate_per_class_lwlrap


class TrainingConfiguration:
    def __init__(self, generator, load_files, num_epoch=200, training_data_dir="processed/melspectrogram/"):
        self.generator = generator
        self.load_files = load_files
        self.num_epoch = num_epoch
        self.training_data_dir = training_data_dir


def calculate_and_dump_lwlrap_per_class(test_file_names, y_test, y_pred, current_fold):
    x_test_fname = np.array([fname[:-7] for fname in test_file_names])

    train_curated = pd.read_csv('data/train_curated.csv')
    train_noisy = pd.read_csv('data/train_noisy.csv')
    single_train = pd.concat([train_curated, train_noisy])
    filter_train_curated = single_train[train_noisy.fname.isin(x_test_fname)]

    labels_count = filter_train_curated['labels'].str.split(expand=True).stack().value_counts()
    labels_count = labels_count.reset_index()
    labels_count.columns = ['class_name', 'sample_count']

    # getting class name
    test = pd.read_csv('data/sample_submission.csv')
    class_names = test.columns[1:]

    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_test, y_pred)
    per_class_lwlrap_df = pd.DataFrame(
        {
            'class_name': class_names,
            'lwlrap': per_class_lwlrap,
            'weighting': weight_per_class
        }
    )
    per_class_lwlrap_df = per_class_lwlrap_df.join(labels_count.set_index('class_name'), on='class_name')
    per_class_lwlrap_df.to_csv("per_class_lwlrap_fold_{}.csv".format(current_fold), index=False)
    print(per_class_lwlrap_df.head())


def reset_keras():
    tf.keras.backend.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


def kfold_validation(train_config: TrainingConfiguration, input_data, input_labels):
    kf = KFold(n_splits=5, shuffle=True)
    fold_scores = []
    current_fold = 1
    start_time = time.time()

    # there is a bug for RTX gpu when I need to set allow_growth to True to run CNN
    reset_keras()

    for train_index, test_index in kf.split(input_data):
        x_train, x_test = input_data[train_index], input_data[test_index]
        y_train, y_test = input_labels.values[train_index], input_labels.values[test_index]

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

        train_generator = train_config.generator(x_train, y_train, batch_size=32, directory=train_config.training_data_dir)
        x, y = train_generator[0]
        max_row = x[0].shape[0]
        max_column = x[0].shape[1]
        max_depth = x[0].shape[2]
        num_classes = y[0].shape[0]
        input_shape = (max_row, max_column, max_depth)
        print("Training shape: ", (max_row, max_column, max_depth))

        x_val = train_config.load_files(x_val, directory=train_config.training_data_dir)

        # create 2d conv model
        model = create_model_simplecnn(input_shape, num_classes)
        # model = resnet_v1(input_shape, 56, num_classes)
        opt = tf.keras.optimizers.Adam()
        model.compile(loss=bce_with_logits, optimizer=opt, metrics=[tf_lwlrap])

        callbacks = [
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

        calculate_and_dump_lwlrap_per_class(x_test, y_test, y_pred, current_fold)

        current_fold += 1
        fold_scores.append(lwlrap)
    print(fold_scores)
    print("Average Fold Score:", np.mean(fold_scores))
    print("Time taken: {}".format(date.compress(time.time() - start_time)))


def train():
    train_curated = pd.read_csv("data/train_noisy.csv")
    file_names = train_curated['fname']
    labels = train_curated['labels'].str.get_dummies(sep=',')
    file_names = np.array([file_name + ".pickle" for file_name in file_names])

    train_config = TrainingConfiguration(
        MelDataGenerator,
        load_melspectrogram_files,
        training_data_dir="processed/melspectrogram_noisy/")

    kfold_validation(train_config, file_names, labels)


if __name__ == "__main__":
    train()
