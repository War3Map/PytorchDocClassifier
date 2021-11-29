import math

import cv2
import pytesseract
import numpy as np
from scipy import ndimage

from image_correction import correct_rotation

TESSERACT_PATH = r"D:\Tesseract\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def show_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image


def prepare_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    img = cv2.bilateralFilter(img, 3, 35, 35)
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []

    show_image(img)
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    show_image(img)
    img = cv2.erode(img, kernel, iterations=2)
    show_image(img)
    skel = img.copy()
    kernel = np.ones((3, 3), np.uint8)
    erod = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    temp = cv2.morphologyEx(erod, cv2.MORPH_DILATE, kernel)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    show_image(img)
    result_image = cv2.threshold(skel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return result_image


def tesseract_recognize(image_path):
    """

    :param image_path:
    :return:
    """
    # img = cv2.imread(image_path)

    image, _, _ = correct_rotation(image_path)

    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_string(image,
                                         config=custom_config,
                                         lang='rus')
    return result


def save_to_txt(string_data, output_filename):
    """
    Save the data file

    :param output_filename: path to output file
    :param string_data: data
    :return:
    """
    with open(output_filename, "w", encoding="utf-8") as data_file:
        data_file.write(string_data)


def recognize_image(source_path, result_path):
    save_to_txt(tesseract_recognize(source_path), result_path)


TEST_IMAGE = r"..."

if __name__ == "__main__":
    recognize_image(TEST_IMAGE, "test_result")
