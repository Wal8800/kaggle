import keras
import numpy as np
import pandas as pd
import librosa


class WaveDataGenerator(keras.utils.Sequence):
    def __init__(self, file_paths, labels=None, batch_size=32, sampling_rate=44100, audio_duration=2, mixup=False, image_aug=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.samples = self.sampling_rate * self.audio_duration

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = np.array([self.read_audio(file_path) for file_path in batch_x])

        if self.labels is None:
            return data

        batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
        return data, batch_y

    def read_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sampling_rate)
        # trim silence
        if 0 < len(y):  # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y)  # trim, top_db=default(60)
        # make it unified length to conf.samples
        if len(y) > self.samples:
            # take the first n samples
            y = y[0:0 + self.samples]
        else:  # pad blank
            padding = self.samples - len(y)  # add padding at both ends
            offset = padding // 2
            y = np.pad(y, (offset, self.samples - len(y) - offset), 'constant')
        return y.reshape((1, self.samples))


    @staticmethod
    def load_wave_files(file_paths):
        generator = WaveDataGenerator(file_paths)
        data = [ generator.read_audio(file_path) for file_path in file_paths]
        return np.array(data)


def test_generate_wave_files():
    train_curated = pd.read_csv('data/train_curated.csv')
    labels = train_curated['labels'].str.get_dummies(sep=',')
    file_paths = np.array(['data/train_curated/' + file_name for file_name in train_curated['fname']])
    generator = WaveDataGenerator(file_paths, labels)
    x, y = generator[0]
    print(x.shape)
    print(y.shape)


if __name__ == "__main__":
    test_generate_wave_files()
