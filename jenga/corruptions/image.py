import numpy as np
import random
import cv2

from jenga.basis import DataCorruption


class GaussianNoise(DataCorruption):

    def __init__(self, fraction, sigma=25):
        self.fraction = fraction
        self.sigma = sigma
        DataCorruption.__init__(self)

    def transform(self, data):
        noisy_images = data.copy()

        for index in range(0, len(noisy_images)):
            if random.random() < self.fraction:
                raw_image = noisy_images[index]
                gaussian = np.random.normal(0, self.sigma, raw_image.shape)
                noisy_image = raw_image + gaussian
                cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
                noisy_image = noisy_image.astype(np.uint8)
                noisy_images[index] = noisy_image

        return noisy_images
