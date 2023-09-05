from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('csvs/clean_data.csv')

print("\n")
print(df.head())

print('number of genre : {}'.format(df.track_genre.nunique()))
print(df.track_genre.unique())

X = df.drop(['track_genre'], axis=1)
y = df['track_genre']

print("\n")
print(X.head())
print("\n")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)


y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print("\n")
print("Y-Train after one-hot encoding: \n")
print(y_train[:10,:])

print("\n")
print("train and test shaope: \n")
print(X_train.shape, X_test.shape)

model = Sequential()
model.add(Dense(12, input_dim=13, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=200, batch_size=20, validation_split=0.1)

score = model.evaluate(X_test, y_test)
print("\n")
print("Loss: ", score[0])
print("Accuracy: ", score[1])
print("\n")

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

img = Image.open('model.png')
img.show()
