import keras
import numpy as np
import librosa
import random


# https://github.com/Cocoxili/DCASE2018Task2
class MelDataGenerator(keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=32, mixup=False, image_aug=False):
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

        if self.labels is None:
            return data

        batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
        return data, batch_y

    @staticmethod
    def augment_melspectrogram(logmel):
        # return logmel
        logmel = logmel.reshape((logmel.shape[0], logmel.shape[1]))
        delta = librosa.feature.delta(logmel)
        accelerate = librosa.feature.delta(logmel, order=2)

        return np.stack((logmel, delta, accelerate), axis=2)


def load_melspectrogram_files(file_paths):
    data = [MelDataGenerator.augment_melspectrogram(np.load(file_path, allow_pickle=True))
            for file_path in file_paths]
    return np.array(data)