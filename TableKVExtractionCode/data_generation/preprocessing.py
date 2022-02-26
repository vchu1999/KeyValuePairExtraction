import cv2
import numpy as np


def ocr_preprocess(img):
    """
    Preprocesses an image
    :param img: image to be processed
    :return: processed image
    """
    out = remove_noise_and_smooth(img)
    thresh = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh


def remove_noise_and_smooth(img):
    """
    Reduces noise and applies smoothening to an image
    :param img: RGB image to be processed
    :return: grayscale processed image
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # converts an image from RGB to grayscale
    gray = cv2.erode(img, np.ones((1, 1), np.uint8), iterations=5)  # performs an erosion

    smoothened = image_smoothening(img)

    # performs an adaptive threshold
    filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)

    kernel = np.ones((1, 1), np.uint8)

    # reduce noise in the background with open operation (erosion than dilation)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

    # reduce noise in the foreground with close operation (dilation than erosion)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # perform bitwise-or operation between smoothened image and processed image
    final_img = cv2.bitwise_or(smoothened, closing)

    return final_img


def image_smoothening(img):
    """
    Applies smoothening to a greyscale image
    :param img: grayscale image to be processed
    :returns: smoothened image
    """
    ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3
