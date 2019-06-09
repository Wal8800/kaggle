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

    for fold_count in range(2, 5):
        additional_fold = pd.read_csv("per_class_lwlrap_fold_{}.csv".format(fold_count))
        fold_one = fold_one.join(additional_fold.set_index("class_name"),
                                 on="class_name",
                                 lsuffix="_{}".format(fold_count))

    fold_one['average_lwlrap'] = fold_one[['lwlrap', 'lwlrap_2', 'lwlrap_3', 'lwlrap_4']].mean(axis=1)
    fold_one['average_weighting'] = fold_one[['weighting', 'weighting_2', 'weighting_3', 'weighting_4']].mean(axis=1)
    print(fold_one.sort_values(by="average_lwlrap", ascending=True).head(20))
    print(fold_one.sort_values(by="average_lwlrap", ascending=False).head())
    print(fold_one['average_lwlrap'].describe())

    result_df = pd.DataFrame(
        {
            'class_name': fold_one['class_name'],
            'lwlrap': fold_one['average_lwlrap'],
            'weighting': fold_one['average_weighting']
        }
    )
    print(result_df.head())

    result_df.to_csv('all_fold_per_class_lwlrap.csv', index=False)


def test_curated_on_noisy_only_model():
    model_dir = "models/"
    function_map = {
        'bce_with_logits': bce_with_logits,
        'tf_lwlrap': tf_lwlrap
    }
    train_curated = pd.read_csv('data/train_curated.csv')
    file_names = train_curated['fname']
    file_names = np.array(["processed/melspectrogram/" + file_name + ".pickle" for file_name in file_names])
    labels = train_curated['labels'].str.get_dummies(sep=',')

    test = pd.read_csv('data/sample_submission.csv')
    class_names = test.columns[1:]
    for fold in range(1, 2):
        tf.keras.backend.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))

        loaded_model = tf.keras.models.load_model(model_dir + "best_{}.h5".format(fold), custom_objects=function_map)
        test_generator = MelDataGenerator(file_names, labels, batch_size=32)

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
        per_class_lwlrap_df.to_csv('noisy_on_curated_per_class_lwlrap.csv', index=False)


def get_best_class(file_path_one, file_path_two):
    # assuming they have same struct, [class_name, lwlrap, weighting]
    lwlrap_one = pd.read_csv(file_path_one)
    lwlrap_two = pd.read_csv(file_path_two)

    lwlrap = lwlrap_one.join(lwlrap_two.set_index("class_name"), on="class_name", rsuffix="_2")
    lwlrap['lwlrap_diff'] = lwlrap['lwlrap'] - lwlrap['lwlrap_2']

    print("Classes that file 1 is better than file 2")
    print(lwlrap.sort_values(by='lwlrap_diff', ascending=False).dropna().head(20))

    print("Classes that file 2 is better than file 1")
    print(lwlrap.sort_values(by='lwlrap_diff', ascending=True).dropna().head(20))

    lwlrap.to_csv("curated_noisy_lwlrap_diff.csv", index=False)


def get_best_gains_for_noisy():
    train_noisy = pd.read_csv('data/train_noisy.csv')
    lwlrap_diff = pd.read_csv("curated_noisy_lwlrap_diff.csv")

    def calculate_gains(class_names):
        total_gains = 0
        for class_name in class_names:
            row = lwlrap_diff[lwlrap_diff.class_name == class_name]
            total_gains += row['lwlrap_diff'].values[0]

        return total_gains / len(class_names)

    train_noisy['gains'] = train_noisy['labels'].apply(lambda x: calculate_gains(str(x).split(sep=',')))

    print(train_noisy.sort_values(by='gains', ascending=True).head(20))
    print(train_noisy[train_noisy.gains < 0].shape)


if __name__ == "__main__":
    all_fold_lwlrap()
    # test_curated_on_noisy_only_model()
    get_best_class("all_fold_per_class_lwlrap.csv", "noisy_on_curated_per_class_lwlrap.csv")
    # get_best_gains_for_noisy()
