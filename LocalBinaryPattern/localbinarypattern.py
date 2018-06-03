import cv2
import numpy as np


def getPixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbpPixelCalculate(img, x, y):
    center = img[x][y]
    val = [getPixel(img, center, x - 1, y + 1),
           getPixel(img, center, x, y + 1),
           getPixel(img, center, x + 1, y + 1),
           getPixel(img, center, x + 1, y),
           getPixel(img, center, x + 1, y - 1),
           getPixel(img, center, x, y - 1),
           getPixel(img, center, x - 1, y - 1),
           getPixel(img, center, x - 1, y)]

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    new_val = 0
    for i in range(len(val)):
        new_val += val[i] * power_val[i]
    return new_val

class LocalBinaryPattern:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        height, width, channel = image.shape
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros((height, width, 3), np.uint8)
        img_lbp = cv2.resize(img_lbp,(width//5, height//5),
        interpolation=self.inter)
        for i in range(0, height//5):
            for j in range(0, width//5):
                img_lbp[i, j] = lbpPixelCalculate(img_gray, i, j)

        return cv2.resize(img_lbp,(self.width, self.height),
        interpolation=self.inter)



