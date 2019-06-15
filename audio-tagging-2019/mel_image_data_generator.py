import keras
import numpy as np
import random
import pandas as pd

from typing import List
from imgaug import augmenters as iaa

import matplotlib.pyplot as plt


def mix_up(x, y):
    x = np.array(x, np.float32)
    lam = np.random.beta(1.0, 1.0)
    ori_index = np.arange(int(len(x)))
    index_array = np.arange(int(len(x)))
    np.random.shuffle(index_array)

    mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
    mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

    return mixed_x, mixed_y


def mix_up_8th(batch_x, batch_y, batch_x_2, batch_y_2):
    lam = np.random.beta(1.0, 1.0)
    mixed_x = lam * batch_x + (1 - lam) * batch_x_2
    mixed_y = lam * batch_y + (1 - lam) * batch_y_2

    return mixed_x, mixed_y


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


augmentation_list = [
    iaa.Fliplr(0.5),
    iaa.CoarseDropout(0.12, size_percent=0.05)
]
image_augmentation = iaa.Sequential(augmentation_list, random_order=True)


class MelDataImageGenerator(keras.utils.Sequence):
    def __init__(self, file_paths: List[str], labels, batch_size=32, mixup=False, image_aug=False):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.mixup = mixup
        self.image_aug = image_aug

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = np.array(
            [MelDataImageGenerator.augment_melspectrogram(np.load(file_path, allow_pickle=True))
                for file_path in batch_x]
        )

        if self.image_aug:
            data = image_augmentation(images=data)

            eraser = get_random_eraser(pixel_level=True)
            data = np.array([eraser(img) for img in data])

        if self.labels is None:
            return data

        batch_y = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
        if self.mixup and random.uniform(0, 1) < 0.25:
            # data, batch_y = mix_up(data, batch_y)

            random_indice = np.random.choice(len(self.file_paths), len(batch_x))
            batch_x_2 = np.array(
                [MelDataImageGenerator.augment_melspectrogram(np.load(file_path, allow_pickle=True))
                    for file_path in self.file_paths[random_indice]]
            )
            if isinstance(self.labels, pd.Series):
                batch_y_2 = self.labels.values[random_indice]
            else:
                batch_y_2 = self.labels[random_indice]
            data, batch_y = mix_up_8th(data, batch_y, batch_x_2, batch_y_2)

        return data, batch_y

    @staticmethod
    def augment_melspectrogram(logmel):
        # return logmel
        if len(logmel.shape) > 2 and logmel.shape[2] == 1:
            logmel = logmel.reshape((logmel.shape[0], logmel.shape[1]))
        return mono_to_color(logmel)


def load_melspectrogram_image_files(file_paths):
    data = [MelDataImageGenerator.augment_melspectrogram(np.load(file_path, allow_pickle=True))
            for file_path in file_paths]
    return np.array(data)


def test_generate_mel_data_files():
    train_curated = pd.read_csv('data/train_curated.csv')
    file_names = train_curated['fname']
    labels = train_curated['labels'].str.get_dummies(sep=',')
    file_paths = np.array(["processed/melspectrogram_128/" + file_name + ".pickle" for file_name in file_names])
    generator = MelDataImageGenerator(file_paths, labels, mixup=False, image_aug=True)

    x, y = generator[0]
    print(x.shape)
    print(y.shape)

    plt.figure()
    plt.imshow(x[0])
    plt.show()

    x = load_melspectrogram_image_files(file_paths[:33])
    print(x.shape)

    plt.figure()
    plt.imshow(x[0])
    plt.show()


if __name__ == "__main__":
    test_generate_mel_data_files()