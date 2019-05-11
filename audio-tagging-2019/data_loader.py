import tensorflow as tf
import numpy as np
import pandas as pd
import librosa


class MelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, fnames, labels, directory, batch_size=32):
        self.fnames = fnames
        self.labels = labels
        self.batch_size = batch_size
        self.directory = directory

    def __len__(self):
        return int(np.ceil(len(self.fnames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.fnames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([np.load(self.directory + file_name + ".pickle", allow_pickle=True) for file_name in batch_x]), np.array(batch_y)

    @staticmethod
    def augment_melspectrogram(logmel):
        delta = librosa.feature.delta(logmel)
        accelerate = librosa.feature.delta(logmel, order=2)

        return np.stack((logmel, delta, accelerate))


def load_melspectrogram_files(directory: str, batch_x):
    return np.array([np.load(directory + file_name + ".pickle", allow_pickle=True) for file_name in batch_x])


if __name__ == "__main__":
    train_curated = pd.read_csv("data/train_curated.csv")
    labels = train_curated['labels'].str.get_dummies(sep=',')
    generator = MelDataGenerator(train_curated["fname"], labels, "processed/melspectrogram/")
    print(len(generator))
    x, y = generator[0]
    print(x[0].shape)
    print(y[0].shape[0])

