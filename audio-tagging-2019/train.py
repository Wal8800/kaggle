import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.metrics

from model import simple_2d_conv
from tensorflow._api.v1.keras.optimizers import SGD
from tensorflow._api.v1.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


# Calculate the overall lwlrap using sklearn.metrics function.
def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap


def reshape_dataframe_to_ndarray(df):
    reshape_arr = []
    for index, row in df.iterrows():
        reshape_arr.append([
            row["mel_spectrogram"]
        ])
    return np.array(reshape_arr)


def train():
    # need to set this configuration to run tensorflow RTX 2070 GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    train_curated = pd.read_csv('data/train_curated.csv')
    labels = train_curated['labels'].str.get_dummies(sep=',')
    train_curated_melspec = pd.read_pickle('padded_train_curated_melspectrogram.pickle')

    # Assume the specotrgram will be the same size after processing the data
    max_row = train_curated_melspec['mel_spectrogram'][0].shape[0]
    max_col = train_curated_melspec['mel_spectrogram'][0].shape[1]

    print('Max row size: ', max_row, 'Max column row size', max_col)
    melspec_ndarray = reshape_dataframe_to_ndarray(train_curated_melspec).reshape((labels.shape[0], max_row, max_col, 1))
    x_train, x_test, y_train, y_test = train_test_split(melspec_ndarray, labels, test_size=0.20)

    print('Train data and label sizes: ', x_train.shape, x_test.shape)
    print('Test data and label sizes: ', y_train.shape, y_test.shape)

    # create 2d conv model
    model = simple_2d_conv((max_row, max_col, 1), len(labels.columns))
    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [
        TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                    write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                    embeddings_data=None, update_freq='epoch')
    ]

    model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=callbacks)
    y_pred = model.predict(x_test)

    lwlrap = calculate_overall_lwlrap_sklearn(y_test.values, y_pred)
    print("Score: ", lwlrap)

    # model_json = model.to_json()
    # with open("./models/model.json", "w") as json_file:
    #     json_file.write(model_json)
    #
    # model.save_weights("./models/model.h5")


if __name__ == "__main__":
    train()
