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


# import numpy as np
# import random
# import cv2
#
# from jenga.basis import DataCorruption
#
#
# class GaussianNoise(DataCorruption):
#
#     def __init__(self, fraction, sigma=25):
#         self.fraction = fraction
#         self.sigma = sigma
#         DataCorruption.__init__(self)
#
#     def transform(self, data):
#         noisy_images = data.copy()
#
#         for index in range(0, len(noisy_images)):
#             if random.random() < self.fraction:
#                 raw_image = noisy_images[index]
#                 gaussian = np.random.normal(0, self.sigma, raw_image.shape)
#                 noisy_image = raw_image + gaussian
#                 cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
#                 noisy_image = noisy_image.astype(np.uint8)
#                 noisy_images[index] = noisy_image
#
#         return noisy_images
#
#
# class Rotation(DataCorruption):
#
#     def __init__(self, fraction):
#         self.fraction = fraction
#
#     def transform(self, data):
#         # we operate on a copy of the data
#         rotated_images = data.copy()
#
#         for index in range(0, len(rotated_images)):
#             if random.random() < self.fraction:
#                 raw_image = rotated_images[index]
#
#                 img = np.zeros([28, 28, 3])
#                 img[:, :, 0] = raw_image.reshape(28, 28)
#                 img[:, :, 1] = raw_image.reshape(28, 28)
#                 img[:, :, 2] = raw_image.reshape(28, 28)
#
#                 degree = np.random.randint(0, 359)
#
#                 rotation = cv2.getRotationMatrix2D((14, 14), degree, 1)
#                 rotated = cv2.warpAffine(img, rotation, (28, 28))
#
#                 rotated_images[index] = rotated[:, :, 0]
#
#         return rotated_images
#
#
# class Crop(DataCorruption):
#
#     def __init__(self, fraction):
#         self.fraction = fraction
#
#     def transform(self, data):
#         # we operate on a copy of the data
#         cropped_images = data.copy()
#
#         for index in range(0, len(cropped_images)):
#             if random.random() < self.fraction:
#                 raw_image = cropped_images[index]
#
#                 start_row = np.random.randint(3) + 2
#                 height = np.random.randint(20) + 3
#                 start_col = np.random.randint(3) + 2
#                 width = np.random.randint(20) + 3
#
#                 # There is probably a vectorized fast version of this
#                 for row in range(start_row, start_row + height):
#                     for col in range(start_col, start_col + width):
#                         raw_image[row, col] = 255
#
#         return cropped_images
