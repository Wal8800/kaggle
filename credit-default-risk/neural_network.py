import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))


model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss="cate")