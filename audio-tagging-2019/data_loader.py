import tensorflow as tf
import numpy as np
import pandas as pd
import librosa


# https://github.com/Cocoxili/DCASE2018Task2
class MelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = [MelDataGenerator.augment_melspectrogram(np.load(file_path, allow_pickle=True))
                for file_path in batch_x]

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


def load_melspectrogram_files(file_paths):
    data = [MelDataGenerator.augment_melspectrogram(np.load(file_path, allow_pickle=True))
            for file_path in file_paths]
    return np.array(data)


class MFCCDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, fnames, labels, directory="processed/mfcss/", batch_size=32):
        self.fnames = fnames
        self.labels = labels
        self.batch_size = batch_size
        self.directory = directory

    def __len__(self):
        return int(np.ceil(len(self.fnames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.fnames[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = [MFCCDataGenerator.augment_mfccs(np.load(self.directory + file_name + ".pickle", allow_pickle=True))
                for file_name in batch_x]

        if self.labels is None:
            return np.array(data)
        else:
            batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array(data), np.array(batch_y)

    @staticmethod
    def augment_mfccs(mfccs):
        mfccs = mfccs.reshape((mfccs.shape[0], mfccs.shape[1]))
        delta = librosa.feature.delta(mfccs)
        accelerate = librosa.feature.delta(mfccs, order=2)

        return np.stack((mfccs, delta, accelerate), axis=2)


def load_mfccs_files(directory: str, batch_x):
    data = [MFCCDataGenerator.augment_mfccs(np.load(directory + file_name + ".pickle", allow_pickle=True))
            for file_name in batch_x]
    return np.array(data)


if __name__ == "__main__":
    train_curated = pd.read_csv("data/train_curated.csv")
    labels = train_curated['labels'].str.get_dummies(sep=',')