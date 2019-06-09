import pandas as pd
import numpy as np
import tensorflow as tf
import time
import keras
import os

from model import create_model_simplecnn, create_model_cnn8th, create_model_resnet, create_model_inception, create_model_cnn_for_wave
from data_loader import MelDataGenerator, load_melspectrogram_files, WaveDataGenerator, load_wave_files, load_melspectrogram_image_files, MelDataImageGenerator
from sklearn.model_selection import KFold
from natural import date
from sklearn.model_selection import train_test_split
from train_util import calculate_overall_lwlrap_sklearn, EarlyStoppingByLWLRAP, bce_with_logits, tf_lwlrap
from train_util import calculate_per_class_lwlrap


class TrainingConfiguration:
    def __init__(self, create_model_func, generator, load_files, num_epoch=200, use_mixup=False, use_img_aug=False,
                 additional_train_data=None):
        self.create_model_func = create_model_func
        self.generator = generator
        self.load_files = load_files
        self.num_epoch = num_epoch
        self.use_mixup = use_mixup
        self.use_img_aug = use_img_aug
        self.additional_train_data = additional_train_data


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def calculate_and_dump_lwlrap_per_class(test_file_path, y_test, y_pred, output_file_name):
    x_test_fname = np.array([fname[fname.rfind("/")+1:-7] for fname in test_file_path])
    train_curated = pd.read_csv('data/train_curated.csv')
    train_noisy = pd.read_csv('data/train_noisy.csv')
    single_train = pd.concat([train_curated, train_noisy])
    filter_train_curated = single_train[single_train.fname.isin(x_test_fname)]

    labels_count = filter_train_curated['labels'].str.split(expand=True, pat=",").stack().value_counts()
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
    per_class_lwlrap_df.to_csv(output_file_name, index=False)
    print(per_class_lwlrap_df.head())


def reset_keras():
    keras.backend.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.set_session(tf.Session(config=config))


def get_input_shape(x):
    max_row = x.shape[0]
    max_column = x.shape[1] if len(x.shape) > 1 else 1
    if len(x.shape) == 2:
        return max_row, max_column

    max_depth = x.shape[2] if len(x.shape) > 2 else 1
    return max_row, max_column, max_depth


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

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)

        if train_config.additional_train_data is not None:
            print("Additional data:", train_config.additional_train_data[0].shape)
            x_train = np.concatenate([x_train, train_config.additional_train_data[0]])
            y_train = np.concatenate([y_train, train_config.additional_train_data[1]])

        train_generator = train_config.generator(x_train, y_train, batch_size=32, mixup=train_config.use_mixup,
                                                 image_aug=train_config.use_img_aug)
        x, y = train_generator[0]
        input_shape = get_input_shape(x[0])

        num_classes = y[0].shape[0]
        print("Training shape: ", input_shape)
        print("Prediction classes", num_classes)
        print("Trainig data size:", x_train.shape, "Validation data:", x_val.shape, "Test data:", x_test.shape)

        x_val = train_config.load_files(x_val)

        learning_rate = 0.001
        reduce_lr_patience = 6
        reduce_lr_factor = 0.8

        # create 2d conv model
        model = train_config.create_model_func(input_shape, num_classes)
        opt = keras.optimizers.Adam(lr=learning_rate)
        lr_metric = get_lr_metric(opt)
        model.compile(loss=bce_with_logits, optimizer=opt, metrics=[tf_lwlrap, lr_metric])

        log_folder_name = './logs/200_fold_{}_{}_{}_{}_{}'.format(
            current_fold,
            learning_rate,
            reduce_lr_patience,
            reduce_lr_factor,
            int(time.time())
        )
        os.mkdir(log_folder_name)
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_folder_name, histogram_freq=0, batch_size=32,
                                        write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                        update_freq='epoch'),
            # EarlyStoppingByLWLRAP(validation_data=(x_val, y_val), patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_tf_lwlrap', patience=reduce_lr_patience, min_lr=1e-5,
                                              factor=reduce_lr_factor, mode='max'),
            keras.callbacks.ModelCheckpoint('./models/best_image_{}.h5'.format(current_fold), monitor='val_tf_lwlrap', verbose=1, save_best_only=True, mode='max')
        ]

        model.fit_generator(train_generator, epochs=train_config.num_epoch, callbacks=callbacks,
                            validation_data=(x_val, y_val), verbose=2)

        # test_generator = train_config.generator(x_test, y_test)
        x_test_data = train_config.load_files(x_test)
        y_pred = model.predict(x_test_data)
        lwlrap = calculate_overall_lwlrap_sklearn(y_test, y_pred)
        print("Fold {} Score: {}".format(current_fold, lwlrap))

        calculate_and_dump_lwlrap_per_class(x_test, y_test, y_pred, "per_class_lwlrap_fold_{}.csv".format(current_fold))
        current_fold += 1
        fold_scores.append(lwlrap)
        break
    print(fold_scores)
    print("Average Fold Score:", np.mean(fold_scores))
    print("Time taken: {}".format(date.compress(time.time() - start_time)))


def train():
    train_curated = pd.read_csv("data/train_curated.csv")
    curated_labels = train_curated['labels'].str.get_dummies(sep=',')
    file_paths = np.array(["processed/melspectrogram/" + file_name + ".pickle" for file_name in train_curated['fname']])

    train_noisy = pd.read_csv("data/train_noisy.csv")
    noisy_class = [
        # 'Trickle_and_dribble',
        # 'Squek',
        # 'Walk_and_footsteps',
        # 'Zipper_(clothing)',
        # 'Toilet_flush',
        # 'Gasp',
        # 'Sigh',
        'Chirp_and_tweet',
        # 'Hiss'
        # 'Shatter'
    ]
    filter_train_noisy = train_noisy[train_noisy.labels.str.contains('|'.join(noisy_class))]
    noisy_labels = np.array(train_noisy['labels'].str.get_dummies(sep=','))
    filter_train_noisy_label = noisy_labels[filter_train_noisy.index]
    noisy_file_paths = np.array(
        ["processed/melspectrogram_noisy_128/" + file_name + ".pickle" for file_name in filter_train_noisy['fname']])

    train_config = TrainingConfiguration(
        create_model_resnet,
        MelDataImageGenerator,
        load_melspectrogram_image_files,
        num_epoch=150,
        use_img_aug=True,
        use_mixup=True,
        # additional_train_data=(noisy_file_paths, filter_train_noisy_label)
    )

    kfold_validation(train_config, file_paths, curated_labels)


def train_wave():
    train_curated = pd.read_csv('data/train_curated.csv')
    labels = train_curated['labels'].str.get_dummies(sep=',')
    file_paths = np.array(['data/train_curated/' + file_name for file_name in train_curated['fname']])

    train_config = TrainingConfiguration(
        create_model_simplecnn,
        WaveDataGenerator,
        load_wave_files,
        num_epoch=200
    )

    kfold_validation(train_config, file_paths, labels)


if __name__ == "__main__":
    train()
    # train_wave()
