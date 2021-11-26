from typing import Optional

import numpy as np
import tensorflow as tf
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
        """
        Class that represents a binary classification task based on a subset of [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
        Predict whether an image is a sneaker (`0`) or an ankle boot (`1`).

        Args:
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
        """

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

    def fit_baseline_model(self, images: Optional[np.array] = None, labels: Optional[np.array] = None) -> PreprocessingDecorator:
        """
        Because data are images, this overrides the default behavior.

        Fit a baseline model. If no data is given (default), it uses the task's train data and creates the attribute `_baseline_model`. \
            If data is given, it trains this data.

        Args:
            images (Optional[np.array], optional): Data to train. Defaults to None.
            labels (Optional[np.array], optional): Labels to train. Defaults to None.

        Raises:
            ValueError: If `images` is given but `labels` not or vice versa

        Returns:
            PreprocessingDecorator: Trained model
        """

        if (images is None and labels is not None) or (images is not None and labels is None):
            raise ValueError("either set both parameters (images, labels) or non")

        use_original_data = images is None

        if images is None:
            images = self.train_data.copy()
            labels = self.train_labels.copy()

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

        model.fit(reshaped_images, keras.utils.to_categorical(labels))
        model = PreprocessingDecorator(model)

        # only set baseline model attribute if it is trained on the original task data
        if use_original_data:
            self._baseline_model = model

        return model
