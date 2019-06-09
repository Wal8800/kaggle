import pandas as pd
import numpy as np
import librosa
import tqdm
import time
import os
import shutil
import random
import keras
import tensorflow as tf

from imgaug import augmenters as iaa
from multiprocessing import Pool

in_kaggle = False
pickle_dir = "processed/test_melspectrogram/"

if in_kaggle:
    test_data_dir = "../input/freesound-audio-tagging-2019/test/"
else:
    test_data_dir = "data/test/"


def mix_up(x, y):
    x = np.array(x, np.float32)
    lam = np.random.beta(1.0, 1.0)
    ori_index = np.arange(int(len(x)))
    index_array = np.arange(int(len(x)))
    np.random.shuffle(index_array)

    mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
    mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

    return mixed_x, mixed_y


def padded_2d_array(two_dim_array, target_frame_length):
    two_dim_array = two_dim_array.reshape((two_dim_array.shape[0], two_dim_array.shape[1], 1))

    # Random offset / Padding
    if two_dim_array.shape[1] > target_frame_length:
        max_offset = two_dim_array.shape[1] - target_frame_length
        offset = np.random.randint(max_offset)
        data = two_dim_array[:, offset:(target_frame_length + offset), :]
    else:
        if target_frame_length > two_dim_array.shape[1]:
            max_offset = target_frame_length - two_dim_array.shape[1]
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(two_dim_array, ((0, 0), (offset, target_frame_length - two_dim_array.shape[1] - offset), (0, 0)),
                      "constant")
    return data


class MelSpectrogramBuilder:
    def __init__(self, frame_weight=80, frame_shift=10, n_mels=64, sampling_rate=44100, audio_duration=2):
        self.n_fft = int(frame_weight / 1000 * sampling_rate)
        self.n_mels = n_mels
        self.fmin = 20
        self.fmax = sampling_rate // 2
        self.frame_weigth = frame_weight
        self.frame_shift = frame_shift
        self.hop_length = int(frame_shift / 1000 * sampling_rate)
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.input_frame_length = int(self.audio_duration * 1000 / self.frame_shift)

    def generate_log_melspectrogram(self, directory, fname):
        try:
            y, sr = librosa.load(directory + fname, sr=self.sampling_rate)
        except Exception as error:
            print(fname)
            raise error

        y, index = librosa.effects.trim(y, hop_length=self.hop_length)
        s = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        return librosa.power_to_db(s)

    def generate_padded_log_melspectrogram(self, directory, fname):
        logmel = self.generate_log_melspectrogram(directory, fname)
        return padded_2d_array(logmel, self.input_frame_length)

    def generate_padded_log_melspectrogram_from_wave(self, y, sr):
        s = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        logmel = librosa.power_to_db(s)
        return padded_2d_array(logmel, self.input_frame_length)


def process_and_save_logmel_test(row):
    fname = row[1][0]
    builder = MelSpectrogramBuilder()
    padded_melspec = builder.generate_padded_log_melspectrogram(test_data_dir, fname)
    padded_melspec.dump(pickle_dir + fname + ".pickle")


def generate_and_save_to_directory_test(df: pd.DataFrame):
    with Pool(2) as p:
        r = list(tqdm.tqdm(p.imap(process_and_save_logmel_test, df.iterrows()), total=len(df)))


augmentation_list = [
    iaa.Fliplr(0.5),
    # iaa.CoarseDropout(0.12, size_percent=0.05)
]
image_augmentation = iaa.Sequential(augmentation_list, random_order=True)


# https://github.com/Cocoxili/DCASE2018Task2
class MelDataGenerator(keras.utils.Sequence):
    def __init__(self, file_paths, labels=None, batch_size=32, mixup=False, image_aug=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.mixup = mixup
        self.image_aug = image_aug

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = np.array(
            [MelDataGenerator.augment_melspectrogram(np.load(file_path, allow_pickle=True))
                for file_path in batch_x]
        )

        if self.image_aug:
            data = image_augmentation(images=data)

        if self.labels is None:
            return data

        batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
        if self.mixup and random.uniform(0, 1) < 0.25:
            data, batch_y = mix_up(data, batch_y)

            # random_indice = np.random.choice(len(self.file_paths), len(batch_x))
            # batch_x_2 = np.array(
            #     [MelDataGenerator.augment_melspectrogram(np.load(file_path, allow_pickle=True))
            #         for file_path in self.file_paths[random_indice]]
            # )
            # if isinstance(self.labels, pd.Series):
            #     batch_y_2 = self.labels.values[random_indice]
            # else:
            #     batch_y_2 = self.labels[random_indice]
            # data, batch_y = mix_up_8th(data, batch_y, batch_x_2, batch_y_2)

        return data, batch_y

    @staticmethod
    def augment_melspectrogram(logmel):
        logmel = logmel.reshape((logmel.shape[0], logmel.shape[1]))
        delta = librosa.feature.delta(logmel)
        accelerate = librosa.feature.delta(logmel, order=2)

        return np.stack((logmel, delta, accelerate), axis=2)


# https://www.kaggle.com/rio114/keras-cnn-with-lwlrap-evaluation/comments
def tf_one_sample_positive_class_precisions(y_true, y_pred):
    num_samples, num_classes = y_pred.shape

    # find true labels
    pos_class_indices = tf.where(y_true > 0)

    # put rank on each element
    retrieved_classes = tf.nn.top_k(y_pred, k=num_classes).indices
    sample_range = tf.zeros(shape=tf.shape(tf.transpose(y_pred)), dtype=tf.int32)
    sample_range = tf.add(sample_range, tf.range(tf.shape(y_pred)[0], delta=1))
    sample_range = tf.transpose(sample_range)
    sample_range = tf.reshape(sample_range, (-1, num_classes * tf.shape(y_pred)[0]))
    retrieved_classes = tf.reshape(retrieved_classes, (-1, num_classes * tf.shape(y_pred)[0]))
    retrieved_class_map = tf.concat((sample_range, retrieved_classes), axis=0)
    retrieved_class_map = tf.transpose(retrieved_class_map)
    retrieved_class_map = tf.reshape(retrieved_class_map, (tf.shape(y_pred)[0], num_classes, 2))

    class_range = tf.zeros(shape=tf.shape(y_pred), dtype=tf.int32)
    class_range = tf.add(class_range, tf.range(num_classes, delta=1))

    class_rankings = tf.scatter_nd(retrieved_class_map,
                                   class_range,
                                   tf.shape(y_pred))

    # pick_up ranks
    num_correct_until_correct = tf.gather_nd(class_rankings, pos_class_indices)

    # add one for division for "presicion_at_hits"
    num_correct_until_correct_one = tf.add(num_correct_until_correct, 1)
    num_correct_until_correct_one = tf.cast(num_correct_until_correct_one, tf.float32)

    # generate tensor [num_sample, predict_rank],
    # top-N predicted elements have flag, N is the number of positive for each sample.
    sample_label = pos_class_indices[:, 0]
    sample_label = tf.reshape(sample_label, (-1, 1))
    sample_label = tf.cast(sample_label, tf.int32)

    num_correct_until_correct = tf.reshape(num_correct_until_correct, (-1, 1))
    retrieved_class_true_position = tf.concat((sample_label,
                                               num_correct_until_correct), axis=1)
    retrieved_pos = tf.ones(shape=tf.shape(retrieved_class_true_position)[0], dtype=tf.int32)
    retrieved_class_true = tf.scatter_nd(retrieved_class_true_position,
                                         retrieved_pos,
                                         tf.shape(y_pred))
    # cumulate predict_rank
    retrieved_cumulative_hits = tf.cumsum(retrieved_class_true, axis=1)

    # find positive position
    pos_ret_indices = tf.where(retrieved_class_true > 0)

    # find cumulative hits
    correct_rank = tf.gather_nd(retrieved_cumulative_hits, pos_ret_indices)
    correct_rank = tf.cast(correct_rank, tf.float32)

    # compute presicion
    precision_at_hits = tf.truediv(correct_rank, num_correct_until_correct_one)

    return pos_class_indices, precision_at_hits


def tf_lwlrap(y_true, y_pred):
    num_samples, num_classes = y_pred.shape
    pos_class_indices, precision_at_hits = (tf_one_sample_positive_class_precisions(y_true, y_pred))
    pos_flgs = tf.cast(y_true > 0, tf.int32)
    labels_per_class = tf.reduce_sum(pos_flgs, axis=0)
    weight_per_class = tf.truediv(tf.cast(labels_per_class, tf.float32),
                                  tf.cast(tf.reduce_sum(labels_per_class), tf.float32))
    sum_precisions_by_classes = tf.zeros(shape=(num_classes), dtype=tf.float32)
    class_label = pos_class_indices[:, 1]
    sum_precisions_by_classes = tf.unsorted_segment_sum(precision_at_hits,
                                                        class_label,
                                                        num_classes)
    labels_per_class = tf.cast(labels_per_class, tf.float32)
    labels_per_class = tf.add(labels_per_class, 1e-7)
    per_class_lwlrap = tf.truediv(sum_precisions_by_classes,
                                  tf.cast(labels_per_class, tf.float32))
    out = tf.cast(tf.tensordot(per_class_lwlrap, weight_per_class, axes=1), dtype=tf.float32)
    return out


def bce_with_logits(y_true, y_pred):
    return keras.backend.mean(keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)


def predict():
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    if in_kaggle:
        input_dir = "../input/freesound-audio-tagging-2019/"
        model_dir = "../input/initial-model-for-audio-2019/"
    else:
        input_dir = "data/"
        model_dir = "models/"

        keras.backend.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=config))

    test = pd.read_csv(input_dir + "sample_submission.csv")
    test_file_paths = np.array(["processed/test_melspectrogram/" + file_name + ".pickle" for file_name in test['fname']])
    generate_and_save_to_directory_test(test)

    fold_prediction = []
    for fold in range(1):
        test_generator = MelDataGenerator(test_file_paths)
        loaded_model = keras.models.load_model(model_dir + "best_{}.h5".format((fold + 1)),
                                                  custom_objects={
                                                      'bce_with_logits': bce_with_logits,
                                                      'tf_lwlrap': tf_lwlrap
                                                  })
        predictions = loaded_model.predict_generator(test_generator)
        fold_prediction.append(predictions)

    average_prediction = np.mean(fold_prediction, axis=0)

    # creating the dataframe to export to csv and rearranging the columns name
    submission_df = pd.DataFrame(average_prediction)
    submission_df['fname'] = test['fname']
    cols = submission_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    submission_df = submission_df[cols]
    submission_df.columns = test.columns.tolist()
    submission_df.to_csv('submission.csv', index=False)
    print(submission_df.head())

    shutil.rmtree("processed/test_melspectrogram/")


if __name__ == "__main__":
    start_time = time.time()
    predict()
    print("Time taken: {}".format(time.time() - start_time))