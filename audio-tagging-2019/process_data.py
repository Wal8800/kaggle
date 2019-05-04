import librosa
import pandas as pd
import numpy as np
import dask.dataframe as dd
import timeit
import time
import natural.date


def generate_log_melspectrogram(fname):
    try:
        y, sr = librosa.load("data/train_curated/" + fname)
    except Exception as error:
        print(fname)
        raise error
    s = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(s, ref=np.max)


def get_melspec_train_curated(dataframe, column_name):
    ddata = dd.from_pandas(dataframe, npartitions=3)
    res = ddata.map_partitions(
        lambda df: df.apply((lambda row: generated_padded_log_melspectrogram('data/train_curated/', row[column_name], 2)),
                            axis=1),
        meta=(None, 'f8')).compute(scheduler='threads')

    return res


def padd_y(y, duration, sr):
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

    return y


def generated_padded_log_melspectrogram(directory, fname, duration):
    try:
        y, sr = librosa.load(directory + fname)
    except Exception as error:
        print(fname)
        raise error

    y = padd_y(y, duration, sr)

    s = librosa.feature.melspectrogram(y=y, sr=sr)
    return librosa.power_to_db(s)


def padded_melspec_train_curated_pickle():
    train_curated = pd.read_csv('data/train_curated.csv')
    melspec = get_melspec_train_curated(train_curated, 'fname')

    melspec_df = pd.DataFrame(
        {
            'mel_spectrogram': melspec,
            'labels': train_curated['labels']
        }
    )
    melspec_df.to_pickle('padded_train_curated_melspectrogram.pickle')


if __name__ == "__main__":
    start_time = time.time()
    padded_melspec_train_curated_pickle()
    print("Time taken: {}".format(natural.date.compress(time.time() - start_time)))
    melspec = pd.read_pickle('padded_train_curated_melspectrogram.pickle')
    print(melspec.head())


