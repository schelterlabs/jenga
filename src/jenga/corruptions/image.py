import random

import imgaug.augmenters as iaa
import numpy as np

from ..basis import DataCorruption


# Base class for image corruptions, for which we rely on the augmentor library
# https://github.com/mdbloice/Augmentor
class ImageCorruption(DataCorruption):

    def __init__(self, fraction, corruptions):
        self.fraction = fraction
        self.corruptions = corruptions
        super().__init__()

    def transform(self, data):
        corrupted_images = data.copy()

        seq = iaa.Sequential(self.corruptions)

        for index in range(0, len(corrupted_images)):
            if random.random() < self.fraction:
                img = corrupted_images[index]

                # ugly, superslow hack for too small images from mnist and fashion mnist...
                # we need to have at least 32x32 pixel images for the corruptions to work...
                if img.shape[0] == 28:
                    wrapper = np.zeros((32, 32), dtype=img.dtype)
                    wrapper[2:30, 2:30] = img

                    corr = seq(images=wrapper)

                    corrupted_images[index] = corr[2:30, 2:30]
                else:
                    seq(images=corrupted_images[index])

        return corrupted_images


class GaussianNoiseCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        super().__init__(fraction, [iaa.imgcorruptlike.GaussianNoise(severity=severity)])


class GlassBlurCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        super().__init__(fraction, [iaa.imgcorruptlike.GlassBlur(severity=severity)])


class SnowCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        super().__init__(fraction, [iaa.imgcorruptlike.Snow(severity=severity)])


class MotionBlurCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        super().__init__(fraction, [iaa.imgcorruptlike.MotionBlur(severity=severity)])


class DefocusBlurCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        super().__init__(fraction, [iaa.imgcorruptlike.DefocusBlur(severity=severity)])


class FogCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        super().__init__(fraction, [iaa.imgcorruptlike.Fog(severity=severity)])


class ContrastCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        super().__init__(fraction, [iaa.imgcorruptlike.Contrast(severity=severity)])


class BrightnessCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        super().__init__(fraction, [iaa.imgcorruptlike.Brightness(severity=severity)])
