from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.compose import ColumnTransformer
from tensorflow import keras

from ..basis import BinaryClassificationTask


class PreprocessingDecorator:

    def __init__(self, model):
        self._baseline_model = model

    def predict_proba(self, images):
        normalized_images = images.astype('float32') / 255
        reshaped_images = normalized_images.reshape(images.shape[0], 28, 28, 1)
        return self._baseline_model.predict(reshaped_images)


# Distinguish images of "ankle boots" from images of "sneakers"
class ShoeCategorizationTask(BinaryClassificationTask):

    def __init__(self, seed):

        sneaker_id = 7
        ankle_boot_id = 9

        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, all_train_labels), (test_images, all_test_labels) = fashion_mnist.load_data()

        # AnkleBoot (class=9) vs Sneaker (class=7)
        train_data = train_images[(all_train_labels == ankle_boot_id) | (all_train_labels == sneaker_id)]

        train_labels = all_train_labels[(all_train_labels == ankle_boot_id) | (all_train_labels == sneaker_id)]
        train_labels = np.where(train_labels == ankle_boot_id, 1, train_labels)
        train_labels = np.where(train_labels == sneaker_id, 0, train_labels)

        test_data = test_images[(all_test_labels == ankle_boot_id) | (all_test_labels == sneaker_id)]

        test_labels = all_test_labels[(all_test_labels == ankle_boot_id) | (all_test_labels == sneaker_id)]
        test_labels = np.where(test_labels == ankle_boot_id, 1, test_labels)
        test_labels = np.where(test_labels == sneaker_id, 0, test_labels)

        super().__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            is_image_data=True,
            seed=seed
        )

    def _get_pipeline_grid_scorer_tuple(self, feature_transformation: ColumnTransformer) -> Tuple[Dict[str, object], Any, Dict[str, Any]]:
        pass

    def fit_baseline_model(self, images: Optional[np.array] = None, labels: Optional[np.array] = None):

        if (images is None and labels is not None) or (images is not None and labels is None):
            raise ValueError("either set both parameters (images, labels) or non")

        if images is None:
            images = self.train_data
            labels = self.train_labels

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        normalized_images = images.astype('float32') / 255
        reshaped_images = normalized_images.reshape(images.shape[0], 28, 28, 1)

        model.fit(reshaped_images, to_categorical(labels))

        return PreprocessingDecorator(model)
