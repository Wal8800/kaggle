import librosa
import pandas as pd
import numpy as np
import time
import natural.date
import random
import tqdm
import librosa.display

from multiprocessing import Pool


def padded_2d_array(two_dim_array, target_frame_length, random_starting_offset=True):
    two_dim_array = two_dim_array.reshape((two_dim_array.shape[0], two_dim_array.shape[1], 1))

    # Random offset / Padding
    if two_dim_array.shape[1] > target_frame_length:
        max_offset = two_dim_array.shape[1] - target_frame_length
        if random_starting_offset:
            offset = np.random.randint(max_offset)
        else:
            offset = 0
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
    def __init__(self, frame_weight=80, frame_shift=10, n_mels=64, sampling_rate=44100, audio_duration=2,
                 random_starting_offset=True):
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
        self.random_starting_offset = random_starting_offset

    def generate_log_melspectrogram(self, directory, fname, transform_wave_func=None):
        try:
            y, sr = librosa.load(directory + fname, sr=self.sampling_rate)
        except Exception as error:
            print(fname)
            raise error

        y, index = librosa.effects.trim(y, hop_length=self.hop_length)

        if transform_wave_func is not None:
            y = transform_wave_func(y, sr)

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

    def generate_padded_log_melspectrogram(self, directory, fname, transform_wave_func=None):
        logmel = self.generate_log_melspectrogram(directory, fname, transform_wave_func)
        return padded_2d_array(logmel, self.input_frame_length, self.random_starting_offset)

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


def process_and_save_logmel_train_curated(row):
    fname = row[1][0]
    builder = MelSpectrogramBuilder(random_starting_offset=False)
    padded_melspec = builder.generate_padded_log_melspectrogram("data/train_curated/", fname)
    padded_melspec.dump("processed/melspectrogram_zero_offset/" + fname + ".pickle")


def process_and_save_logmel_train_noisy(row):
    fname = row[1][0]
    builder = MelSpectrogramBuilder()
    padded_melspec = builder.generate_padded_log_melspectrogram("data/train_noisy/", fname)
    padded_melspec.dump("processed/melspectrogram_noisy/" + fname + ".pickle")


# https://www.kaggle.com/daisukelab/fat2019_prep_mels1
def create_preprocessed_mel_builder():
    builder = MelSpectrogramBuilder()
    builder.sampling_rate = 44100
    builder.audio_duration = 3
    builder.hop_length = 347 * builder.audio_duration
    builder.fmin = 20
    builder.fmax = builder.sampling_rate // 2
    builder.n_mels = 128
    builder.n_fft = builder.n_mels * 28
    builder.input_frame_length = 64 * builder.audio_duration
    return builder


def process_and_save_logmel_train_curated_192(row):
    fname = row[1][0]
    builder = create_preprocessed_mel_builder()
    padded_melspec = builder.generate_padded_log_melspectrogram("data/train_curated/", fname)
    padded_melspec.dump("processed/melspectrogram_192/" + fname + ".pickle")


def process_and_save_logmel_train_noisy_128(row):
    fname = row[1][0]
    builder = create_preprocessed_mel_builder()
    padded_melspec = builder.generate_padded_log_melspectrogram("data/train_noisy/", fname)
    padded_melspec.dump("processed/melspectrogram_noisy_128/" + fname + ".pickle")


# https://www.kaggle.com/huseinzol05/sound-augmentation-librosa#apply-hpss
def process_augment_save_logmel(row):
    output_dir = "processed/aug_melspectrogram/"

    fname = row[1][0]
    builder = MelSpectrogramBuilder()

    def transform_wave(y, sr):

        speed_changes = random.uniform(0, 1)

        if speed_changes < 0.25:
            slow_changes = np.random.uniform(low=0.8, high=0.9)
            y = librosa.effects.time_stretch(y, slow_changes)

        if 0.25 <= speed_changes < 0.5:
            fast_changes = np.random.uniform(low=1.1, high=1.2)
            y = librosa.effects.time_stretch(y, fast_changes)

        if random.uniform(0, 1) < 0.25:
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            y = y + noise_amp * np.random.normal(size=y.shape[0])

        if random.uniform(0, 1) < 0.25:
            bins_per_octave = 12
            pitch_pm = 2
            pitch_change = pitch_pm * 2 * (np.random.uniform())
            y = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)

        return y

    padded_melspec = builder.generate_padded_log_melspectrogram("data/train_curated/", fname, transform_wave)
    padded_melspec.dump(output_dir + fname + ".pickle")


def generate_and_save_to_directory(df: pd.DataFrame, process_func):
    with Pool(2) as p:
        r = list(tqdm.tqdm(p.imap(process_func, df.iterrows()), total=len(df)))


def generate_train_curated():
    start_time = time.time()
    train_curated = pd.read_csv('data/train_curated.csv')
    generate_and_save_to_directory(train_curated, process_augment_save_logmel)
    print("Time taken: {}".format(natural.date.compress(time.time() - start_time)))


def generate_train_noisy():
    start_time = time.time()
    train_noisy = pd.read_csv('data/train_noisy.csv')
    generate_and_save_to_directory(train_noisy, process_and_save_logmel_train_noisy)
    print("Time taken: {}".format(natural.date.compress(time.time() - start_time)))


def check_number_of_sample_duration():
    train_curated = pd.read_csv('data/train_curated.csv')
    tc_count = 0
    for file_name in train_curated['fname'].values:
        duration = librosa.get_duration(filename="data/train_curated/" + file_name)
        if duration > 4:
            tc_count += 1

    print("Number of train curated sample with greater 2 second duration:", tc_count)
    print("Total train_curated sample count:", len(train_curated['fname'].values))

    test = pd.read_csv('data/sample_submission.csv')
    test_count = 0
    for file_name in test['fname'].values:
        duration = librosa.get_duration(filename="data/test/" + file_name)
        if duration > 4:
            test_count += 1

    print("Number of test sample with greater 2 second duration:", test_count)
    print("Total train_curated sample count:", len(test['fname'].values))


if __name__ == "__main__":
    generate_train_curated()
