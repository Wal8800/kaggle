import tensorflow as tf
from tensorflow._api.v1.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow._api.v1.keras.optimizers import SGD


def simple_2d_conv(input_shape, num_classes) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == "__main__":
    model = simple_2d_conv((100, 100, 3), 10)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print(model.summary())
