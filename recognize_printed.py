import math

import cv2
import pytesseract
import numpy as np
from scipy import ndimage

TESSERACT_PATH = r"D:\Tesseract\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def prepare_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    img = cv2.bilateralFilter(img, 3, 35, 35)
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=2)
    skel = img.copy()
    kernel = np.ones((3, 3), np.uint8)
    erod = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    temp = cv2.morphologyEx(erod, cv2.MORPH_DILATE, kernel)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    result_image = cv2.threshold(skel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return result_image


# проверка угла изображения, выравнивание
def correct_rotation(imag):
    img_before = imag

    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
                            100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(img_before, median_angle)

    print(f"Angle is {median_angle:.04f}")
    return img_rotated


def tesseract_recognize(image_path):
    """

    :param image_path:
    :return:
    """
    img = cv2.imread(image_path)
    img = correct_rotation(prepare_image(img))
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_string(img, config=custom_config, lang='rus')
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
