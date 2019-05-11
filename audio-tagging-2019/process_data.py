import librosa
import pandas as pd
import numpy as np
import dask.dataframe as dd
import time
import natural.date
from multiprocessing import Pool
import tqdm


class MelSpectrogramBuilder:
    def __init__(self, frame_weight=80, frame_shift=10, n_mels=128, sampling_rate=44100, audio_duration=2):
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

    def get_melspec_tested(self):
        test = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')
        ddata = dd.from_pandas(test, npartitions=3)

        res = ddata.map_partitions(
            lambda df: df.apply((lambda row: self.generate_padded_log_melspectrogram(
                '../input/freesound-audio-tagging-2019/test/', row.fname)), axis=1), meta=(None, 'f8')).compute(
            scheduler='threads')

        return res


def process_and_save_logmel(row):
    fname = row[1][0]
    builder = MelSpectrogramBuilder()
    padded_melspec = builder.generate_padded_log_melspectrogram("data/train_curated/", fname)
    padded_melspec.dump("processed/melspectrogram/" + fname + ".pickle")


def generate_and_save_to_directory(df: pd.DataFrame):
    with Pool(2) as p:
        r = list(tqdm.tqdm(p.imap(process_and_save_logmel, df.iterrows()), total=len(df)))


if __name__ == "__main__":
    start_time = time.time()
    train_curated = pd.read_csv('data/train_curated.csv')
    generate_and_save_to_directory(train_curated)
    print("Time taken: {}".format(natural.date.compress(time.time() - start_time)))
    # builder = MelSpectrogramBuilder()
    # gram = builder.generate_padded_log_melspectrogram("data/train_curated/", "0a9bebde.wav")
    # print(gram.shape)

