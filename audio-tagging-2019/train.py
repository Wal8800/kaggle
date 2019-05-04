import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.metrics
import time

from model import simple_2d_conv
from tensorflow._api.v1.keras.optimizers import SGD
from sklearn.model_selection import KFold
from natural import date


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


def reshape_dataframe_to_ndarray(df, column_name):
    reshape_arr = []
    for index, row in df.iterrows():
        reshape_arr.append([
            row[column_name]
        ])
    return np.array(reshape_arr)


def train():
    # need to set this configuration to run tensorflow RTX 2070 GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    train_curated_melspec = pd.read_pickle('padded_train_curated_melspectrogram.pickle')
    labels = train_curated_melspec['labels'].str.get_dummies(sep=',')

    # Assume the specotrgram will be the same size after processing the data
    max_row = train_curated_melspec['mel_spectrogram'][0].shape[0]
    max_col = train_curated_melspec['mel_spectrogram'][0].shape[1]

    print('Max row size: ', max_row, 'Max column row size', max_col)
    melspec_ndarray = reshape_dataframe_to_ndarray(train_curated_melspec, "mel_spectrogram").reshape(
        (labels.shape[0], max_row, max_col, 1))

    kf = KFold(n_splits=5, shuffle=True)
    fold_scores = []
    current_fold = 1
    start_time = time.time()
    for train_index, test_index in kf.split(melspec_ndarray):
        x_train, x_test = melspec_ndarray[train_index], melspec_ndarray[test_index]
        y_train, y_test = labels.values[train_index], labels.values[test_index]

        # create 2d conv model
        model = simple_2d_conv((max_row, max_col, 1), len(labels.columns))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir='./logs/fold_{}'.format(current_fold), histogram_freq=0,
                                           batch_size=32, write_graph=True, write_grads=False, write_images=False,
                                           embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                                           embeddings_data=None, update_freq='epoch'),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10),
            # tf.keras.callbacks.ModelCheckpoint('./models/best_{}.h5'.format(current_fold),
            #                                    monitor='val_loss', verbose=1, save_best_only=True)
        ]

        model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=callbacks, validation_split=0.2)
        y_pred = model.predict(x_test)

        lwlrap = calculate_overall_lwlrap_sklearn(y_test, y_pred)
        print("Fold {} Score: {}".format(current_fold, lwlrap))
        current_fold += 1
        fold_scores.append(lwlrap)
    print("Average Fold Score:", np.mean(fold_scores))
    print("Time taken: {}".format(date.compress(time.time() - start_time)))


if __name__ == "__main__":
    train()
