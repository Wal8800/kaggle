import kapre
import keras

from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, AveragePooling2D, \
    BatchNormalization, Activation, Input, Convolution2D, PReLU
from keras import regularizers
from tensorflow._api.v1.keras.optimizers import SGD


def _conv_simple_block(x, n_filters):
    x = Convolution2D(n_filters, (3, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution2D(n_filters, (3, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D()(x)

    return x


def create_model_simplecnn(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x = _conv_simple_block(inp, 64)
    x = _conv_simple_block(x, 128)
    x = _conv_simple_block(x, 256)
    x = _conv_simple_block(x, 512)

    x1 = keras.layers.GlobalAveragePooling2D()(x)
    x2 = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Add()([x1, x2])

    x = Dropout(0.2)(x)
    x = Dense(128, activation='linear')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    predictions = Dense(num_classes, activation='linear')(x)

    return keras.models.Model(inputs=inp, outputs=predictions)


# from the 8th solution in 2018 competition
# https://github.com/sainathadapa/kaggle-freesound-audio-tagging
def create_model_cnn8th(input_shape, num_classes):
    regu = 0
    inp = Input(shape=input_shape)

    x = Conv2D(48, 11, strides=(1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(regu))(inp)
    x = BatchNormalization()(x)
    x = Conv2D(48, 11, strides=(2, 3), kernel_initializer='he_uniform', activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(regu))(x)
    x = MaxPooling2D(3, strides=(1, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, 5, strides=(1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(regu))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 5, strides=(2, 3), kernel_initializer='he_uniform', activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(regu))(x)
    x = MaxPooling2D(3, strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(192, 3, strides=1, kernel_initializer='he_uniform', activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, 3, strides=1, kernel_initializer='he_uniform', activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, strides=1, kernel_initializer='he_uniform', activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(regu))(x)
    x = MaxPooling2D(3, strides=(1, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(num_classes, activation='linear')(x)

    return keras.models.Model(inputs=inp, outputs=predictions)


def create_model_resnet(input_shape, num_classes):
    input = Input(shape=input_shape)
    base_model = keras.applications.resnet50.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=input,
        input_shape=input_shape)

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(384, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes, activation='linear')(x)

    return keras.models.Model(inputs=input, outputs=x)


def create_model_inception(input_shape, num_classes):
    input = Input(shape=input_shape)
    base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=input,
        input_shape=input_shape)

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = Dense(1536, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(384, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes, activation='linear')(x)

    return keras.models.Model(inputs=input, outputs=x)


def create_model_cnn_for_wave(input_shape, num_classes):
    sr = 44100
    input = Input(shape=input_shape)

    x = kapre.time_frequency.Melspectrogram(n_dft=512, n_hop=256, input_shape=input_shape, padding='same', sr=sr,
                                            n_mels=128, fmin=0.0, fmax=sr/2, power_melgram=1.0,
                                            return_decibel_melgram=False, trainable_fb=False, trainable_kernel=False,
                                            name='trainable_stft')(input)

    x = kapre.augmentation.AdditiveNoise(power=0.2)(x)
    x = kapre.utils.Normalization2D(str_axis='freq')(x)

    x = _conv_simple_block(x, 64)
    x = _conv_simple_block(x, 128)
    x = _conv_simple_block(x, 256)
    x = _conv_simple_block(x, 512)

    x1 = keras.layers.GlobalAveragePooling2D()(x)
    x2 = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Add()([x1, x2])

    x = Dropout(0.2)(x)
    x = Dense(128, activation='linear')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    predictions = Dense(num_classes, activation='linear')(x)

    return keras.models.Model(inputs=input, outputs=predictions)


if __name__ == "__main__":
    print(create_model_simplecnn((64, 200, 3), 80).count_params())
    # print(create_model_cnn8th((64, 200, 3), 80).count_params())
    # print(create_model_inception((128, 128, 1), 80).count_params())
    # create_model_inception((128, 128, 1), 80).summary()
    create_model_simplecnn((64, 200, 3), 80).summary()

