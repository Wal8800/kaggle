import pandas as pd
import tensorflow as tf
import numpy as np

from train_util import bce_with_logits, tf_lwlrap, calculate_per_class_lwlrap
from data_loader import MelDataGenerator

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def fold_one_lwlrap():
    lwlrap_per_clas = pd.read_csv("per_class_lwlrap_fold_1.csv")

    print(lwlrap_per_clas.head())
    print("Number of class don't have any sample: ", lwlrap_per_clas.isnull().sum().sum())
    filter_lwlrap_per_class = lwlrap_per_clas.dropna(subset=['sample_count'])
    print("Highest lwlrap without Nan")
    print(filter_lwlrap_per_class.sort_values(by='lwlrap', ascending=False).head(10))
    print("Lowest lwlrap without Nan")
    print(filter_lwlrap_per_class.sort_values(by='lwlrap', ascending=True).head(10))


def all_fold_lwlrap():
    fold_one = pd.read_csv("per_class_lwlrap_fold_1.csv")

    for fold_count in range(2, 6):
        additional_fold = pd.read_csv("per_class_lwlrap_fold_{}.csv".format(fold_count))
        fold_one = fold_one.join(additional_fold.set_index("class_name"),
                                 on="class_name",
                                 lsuffix="_{}".format(fold_count))

    fold_one['average_lwlrap'] = fold_one[['lwlrap', 'lwlrap_2', 'lwlrap_3', 'lwlrap_4', 'lwlrap_5']].mean(axis=1)
    print(fold_one.sort_values(by="average_lwlrap", ascending=True).head(20))
    print(fold_one.sort_values(by="average_lwlrap", ascending=False).head())
    print(fold_one['average_lwlrap'].describe())


def test_curated_on_noisy_only_model():
    model_dir = "models/"
    function_map = {
        'bce_with_logits': bce_with_logits,
        'tf_lwlrap': tf_lwlrap
    }
    train_curated = pd.read_csv('data/train_curated.csv')
    file_names = train_curated['fname']
    file_names = np.array([file_name + ".pickle" for file_name in file_names])
    labels = train_curated['labels'].str.get_dummies(sep=',')

    test = pd.read_csv('data/sample_submission.csv')
    class_names = test.columns[1:]
    for fold in range(1, 6):
        tf.keras.backend.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))

        loaded_model = tf.keras.models.load_model(model_dir + "best_{}.h5".format(fold), custom_objects=function_map)
        test_generator = MelDataGenerator(file_names, labels, batch_size=32, directory="processed/melspectrogram/")

        predictions = loaded_model.predict_generator(test_generator)
        per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(np.array(labels), predictions)
        per_class_lwlrap_df = pd.DataFrame(
            {
                'class_name': class_names,
                'lwlrap': per_class_lwlrap,
                'weighting': weight_per_class
            }
        )
        print(per_class_lwlrap_df.sort_values(by='lwlrap', ascending=False).head())
        print(per_class_lwlrap_df.sort_values(by='lwlrap', ascending=True).head())


if __name__ == "__main__":
    test_curated_on_noisy_only_model()
