import imgaug.augmenters as iaa

from jenga.basis import DataCorruption
import numpy as np
import random


class ImageCorruption(DataCorruption):

    def __init__(self, fraction, corruptions):
        self.fraction = fraction
        self.corruptions = corruptions
        DataCorruption.__init__(self)

    def transform(self, data):
        corrupted_images = data.copy()

        seq = iaa.Sequential(self.corruptions)

        for index in range(0, len(corrupted_images)):
            if random.random() < self.fraction:
                img = corrupted_images[index]
                # superslow hack for too small images...
                wrapper = np.zeros((32, 32), dtype=img.dtype)
                wrapper[2:30, 2:30] = img

                corr = seq(images=wrapper)

                corrupted_images[index] = corr[2:30, 2:30]

        return corrupted_images


class GaussianNoiseCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        ImageCorruption.__init__(self, fraction, [iaa.imgcorruptlike.GaussianNoise(severity=severity)])


class GlassBlurCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        ImageCorruption.__init__(self, fraction, [iaa.imgcorruptlike.GlassBlur(severity=severity)])


class SnowCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        ImageCorruption.__init__(self, fraction, [iaa.imgcorruptlike.Snow(severity=severity)])


class MotionBlurCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        ImageCorruption.__init__(self, fraction, [iaa.imgcorruptlike.MotionBlur(severity=severity)])


class DefocusBlurCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        ImageCorruption.__init__(self, fraction, [iaa.imgcorruptlike.DefocusBlur(severity=severity)])


class FogCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        ImageCorruption.__init__(self, fraction, [iaa.imgcorruptlike.Fog(severity=severity)])


class ContrastCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        ImageCorruption.__init__(self, fraction, [iaa.imgcorruptlike.Contrast(severity=severity)])


class BrightnessCorruption(ImageCorruption):

    def __init__(self, fraction, severity):
        ImageCorruption.__init__(self, fraction, [iaa.imgcorruptlike.Brightness(severity=severity)])
