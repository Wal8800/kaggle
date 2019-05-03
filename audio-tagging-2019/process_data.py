import librosa
import pandas as pd
import numpy as np
import dask.dataframe as dd
import timeit
import time


def generate_log_melspectrogram(fname):
    try:
        y, sr = librosa.load("data/train_curated/" + fname)
    except Exception as error:
        print(fname)
        raise error
    s = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(s, ref=np.max)


def get_melspec_train_curated():
    train_curated = pd.read_csv('data/train_curated.csv')
    print(train_curated.head())
    ddata = dd.from_pandas(train_curated, npartitions=3)

    res = ddata.map_partitions(
        lambda df: df.apply((lambda row: generated_padded_log_melspectrogram(row.fname, 2)), axis=1), meta=(None, 'f8')).compute(scheduler='threads')

    print(res.shape)
    return res


def generated_padded_log_melspectrogram(fname, duration):
    try:
        y, sr = librosa.load('data/train_curated/' + fname)
    except Exception as error:
        print(fname)
        raise error
    audio_length = duration * sr
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
    s = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(s, ref=np.max)


def get_and_save_melspec_train_curated_pickle():
    start_time = time.time()
    melspec = get_melspec_train_curated()
    print(type(melspec))
    print(time.time() - start_time)
    melspec_df = pd.DataFrame({'mel_spectrogram': melspec})
    # melspec_df.to_csv('train_curated_melspectrogram.csv', index=False)
    melspec_df.to_pickle('padded_train_curated_melspectrogram.pickle')


if __name__ == "__main__":
    # get_and_save_melspec_train_curated_pickle()
    melspec = pd.read_pickle('padded_train_curated_melspectrogram.pickle')
    reshape_arr = []
    print(melspec.shape)
    print(melspec['mel_spectrogram'][0].shape)
    for index, row in melspec.iterrows():
        reshape_arr.append([
            row["mel_spectrogram"]
        ])
    reshape_arr = np.array(reshape_arr)
    print(reshape_arr.shape)

