import tensorflow as tf
import pandas as pd
import numpy as np
import librosa
import dask.dataframe as dd
from multiprocessing import Pool
import tqdm
import time
import os
import shutil

pickle_dir = "processed/test_melspectrogram/"
# test_data_dir = "../input/freesound-audio-tagging-2019/test/"
test_data_dir = "data/test/"


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

    def generate_padded_melspec_df_pickle(self, df, colunmn_name, file_name):
        melspec = self._get_melspec_from_dataframe(df, colunmn_name)
        melspec_df = pd.DataFrame(
            {
                'mel_spectrogram': melspec,
                'labels': df['labels'],
                'fname': df['fname']
            }
        )
        melspec_df.to_pickle(file_name)

    def _get_melspec_from_dataframe(self, dataframe, column_name):
        ddata = dd.from_pandas(dataframe, npartitions=3)
        res = ddata.map_partitions(
            lambda df: df.apply(
                (lambda row: self.generate_padded_log_melspectrogram('data/train_curated/', row[column_name])),
                axis=1),
            meta=(None, 'f8')).compute(scheduler='threads')

        return res

    def generate_log_melspectrogram(self, directory, fname):
        try:
            y, sr = librosa.load(directory + fname, sr=self.sampling_rate)
        except Exception as error:
            print(fname)
            raise error

        s = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        return librosa.power_to_db(s)

    def generate_padded_log_melspectrogram(self, directory, fname):
        logmel = self.generate_log_melspectrogram(directory, fname)
        return self._padded_logmel(logmel)

    def _padded_logmel(self, logmel):
        logmel = logmel.reshape((logmel.shape[0], logmel.shape[1], 1))
        # print(logmel.shape)
        input_frame_length = int(self.audio_duration * 1000 / self.frame_shift)
        # Random offset / Padding
        if logmel.shape[1] > input_frame_length:
            max_offset = logmel.shape[1] - input_frame_length
            offset = np.random.randint(max_offset)
            data = logmel[:, offset:(input_frame_length + offset), :]
        else:
            if input_frame_length > logmel.shape[1]:
                max_offset = input_frame_length - logmel.shape[1]
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(logmel, ((0, 0), (offset, input_frame_length - logmel.shape[1] - offset), (0, 0)), "constant")
        return data


def process_and_save_logmel_test(row):
    fname = row[1][0]
    builder = MelSpectrogramBuilder()
    padded_melspec = builder.generate_padded_log_melspectrogram(test_data_dir, fname)
    padded_melspec.dump(pickle_dir + fname + ".pickle")


def generate_and_save_to_directory_test(df: pd.DataFrame):
    with Pool(2) as p:
        r = list(tqdm.tqdm(p.imap(process_and_save_logmel_test, df.iterrows()), total=len(df)))


# https://github.com/Cocoxili/DCASE2018Task2
class MelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, fnames, directory, labels=None, batch_size=32):
        self.fnames = fnames
        self.labels = labels
        self.batch_size = batch_size
        self.directory = directory

    def __len__(self):
        return int(np.ceil(len(self.fnames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.fnames[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = [
            MelDataGenerator.augment_melspectrogram(np.load(self.directory + file_name + ".pickle", allow_pickle=True))
            for file_name in batch_x]

        if self.labels is None:
            return np.array(data)
        else:
            batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array(data), np.array(batch_y)

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
    return tf.keras.backend.mean(tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)


def predict():
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    input_dir = "data/"
    # input_dir = "../input/freesound-audio-tagging-2019/"
    test = pd.read_csv(input_dir + "sample_submission.csv")
    generate_and_save_to_directory_test(test)

    # model_dir = "../input/initial-model-for-audio-2019/"
    model_dir = "models/"

    tf.keras.backend.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    fold_prediction = []
    for fold in range(3):
        test_generator = MelDataGenerator(test["fname"], directory=pickle_dir)
        loaded_model = tf.keras.models.load_model(model_dir + "best_{}.h5".format((fold + 1)),
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