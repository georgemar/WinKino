# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from numpy import genfromtxt
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = genfromtxt('train_data.csv', delimiter=",")
test_data = genfromtxt('test_data.csv', delimiter=",")
train_labels = genfromtxt('train_labels.csv', delimiter=",")
test_labels = genfromtxt('test_labels.csv', delimiter=",")

print("Train data sample : {} from the start".format(train_data[0]))
print("Train data sample : {} from the end".format(train_data[-1]))
print("Train labels sample : {} from the start".format(train_labels[0]))
print("Train labels sample : {} from the end".format(train_labels[-1]))
print("Train data length {}".format(len(train_data)))

model = keras.Sequential()
model.add(keras.layers.Embedding(702697, 2))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(50, activation=tf.nn.sigmoid))
model.add(keras.layers.Dense(50, activation=tf.nn.relu))
model.add(keras.layers.Dense(80, activation=tf.nn.relu))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['acc'])


print("Fitting the model")

history = model.fit(train_data,
                    train_labels,
                    epochs=10,
                    batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose=1)

predictions = model.predict(test_data)

for i in range(0, 80):
    print(test_labels[0][i] + " -> " + predictions[0][i])

