import librosa
import pandas as pd
import numpy as np
import dask.dataframe as dd
import time
import natural.date


class MelSpectrogramBuilder:
    def __init__(self, frame_weight=80, frame_shift=10, n_mels=64, sampling_rate=44100, audio_duration=2):
        self.n_fft = int(frame_weight / 1000 * sampling_rate)
        self.n_mels = n_mels
        self.frame_weigth = frame_weight
        self.frame_shift = frame_shift
        self.hop_length = int(frame_shift / 1000 * sampling_rate)
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration

    def generate_padded_melspec_df_pickle(self, file_name):
        train_curated = pd.read_csv('data/train_curated.csv')
        melspec = self._get_melspec_from_dataframe(train_curated, 'fname')
        melspec_df = pd.DataFrame(
            {
                'mel_spectrogram': melspec,
                'labels': train_curated['labels'],
                'fname': train_curated['fname']
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

        s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        return librosa.power_to_db(s)

    def generate_padded_log_melspectrogram(self, directory, fname):
        logmel = self.generate_log_melspectrogram(directory, fname)
        return self.padded_logmel(logmel)

    def pad_wave(self, y):
        audio_length = self.audio_duration * self.sampling_rate

        if len(y) > audio_length:
            max_offset = len(y) - audio_length
            offset = np.random.randint(max_offset)
            y = y[offset:(audio_length + offset)]
        else:
            if audio_length > len(y):
                max_offset = audio_length - len(y)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
        y = np.pad(y, (offset, audio_length - len(y) - offset), "constant")
        return y

    def padded_logmel(self, logmel):
        # print(logmel.shape)
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

    def get_melspec_tested(self):
        test = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')
        ddata = dd.from_pandas(test, npartitions=3)

        res = ddata.map_partitions(
            lambda df: df.apply((lambda row: self.generate_padded_log_melspectrogram(
                '../input/freesound-audio-tagging-2019/test/', row.fname)), axis=1), meta=(None, 'f8')).compute(
            scheduler='threads')

        return res


if __name__ == "__main__":
    builder = MelSpectrogramBuilder()
    padded_gram = builder.generate_padded_log_melspectrogram('data/train_curated/', '0a9bebde.wav')
    print(padded_gram.shape)
    start_time = time.time()
    builder.generate_padded_melspec_df_pickle('padded_train_curated_melspectrogram.pickle')
    print("Time taken: {}".format(natural.date.compress(time.time() - start_time)))
    melspec = pd.read_pickle('padded_train_curated_melspectrogram.pickle')
    print(melspec.head())
    print(melspec.shape)
    print(melspec['mel_spectrogram'][0].shape)
    print(melspec['mel_spectrogram'][0])
    print(melspec['mel_spectrogram'][1].shape)
    print(melspec['mel_spectrogram'][1])
