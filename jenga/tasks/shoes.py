import tensorflow as tf
import numpy as np
from tensorflow import keras

from sklearn.metrics import accuracy_score
from keras.utils import to_categorical


class PreprocessingDecorator:

    def __init__(self, model):
        self.model = model

    def predict_classes(self, images):
        normalized_images = images.astype('float32') / 255
        reshaped_images = normalized_images.reshape(images.shape[0], 28, 28, 1)
        return self.model.predict_classes(reshaped_images)


class ShoeCategorizationTask:

    def __init__(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (all_train_images, all_train_labels), (all_test_images, all_test_labels) = fashion_mnist.load_data()

        # Sneaker (class=7) vs AnkleBoot (class=9)
        self.train_images = all_train_images[(all_train_labels == 9) | (all_train_labels == 7)]

        train_labels = all_train_labels[(all_train_labels == 9) | (all_train_labels == 7)]
        train_labels = np.where(train_labels == 9, 1, train_labels)
        self.train_labels = np.where(train_labels == 7, 0, train_labels)

        self.test_images = all_test_images[(all_test_labels == 9) | (all_test_labels == 7)]
        test_labels = all_test_labels[(all_test_labels == 9) | (all_test_labels == 7)]
        test_labels = np.where(test_labels == 9, 1, test_labels)
        self.__test_labels = np.where(test_labels == 7, 0, test_labels)

    def fit_baseline_model(self, images, labels):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                         input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        normalized_images = images.astype('float32') / 255
        reshaped_images = normalized_images.reshape(images.shape[0], 28, 28, 1)

        model.fit(reshaped_images, to_categorical(labels))

        return PreprocessingDecorator(model)

    def score_on_test_images(self, predicted_labels):
        return accuracy_score(self.__test_labels, predicted_labels)
