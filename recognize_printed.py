from time import time

import cv2
import pytesseract
from pytesseract import Output

from image_correction import correct_rotation
from settings import TESSERACT_PATH, RECOGNIZE_TEST_IMAGE

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def show_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image


def tesseract_recognize(image_path):
    """

    :param image_path:
    :return:
    """
    image = cv2.imread(image_path)

    # correct rotation, align text
    start_time = time()
    image, _, _ = correct_rotation(image)
    end_time = time() - start_time
    print(f"Корректировка угла: {end_time}")
    show_image(image)

    # detect text rotation angle
    start_time = time()
    info = pytesseract.image_to_osd(image, output_type=Output.DICT)
    # print(info)
    end_time = time() - start_time
    angle = info["orientation"]
    print(f"Определение угла({angle}): {end_time}")

    # correct rotation angle in case of 270
    if angle == 270:
        angle = 90
    # rotate text if it upside down or 90 deg
    if angle in (90, 180):
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        # Perform the rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (w, h))
        print(f"Поворот:!")
        show_image(image)

    # Adding custom options
    # Page segmentation mode
    # 6 = Assume a single uniform block of text.
    # OCR Engine modes:
    # 3 = Default, based on what is available.

    custom_config = r'--oem 3 --psm 6'
    start_time = time()
    result = pytesseract.image_to_string(image,
                                         config=custom_config,
                                         lang='rus')
    end_time = time() - start_time
    print(f"Распознавание: {end_time}")
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


if __name__ == "__main__":
    recognize_image(RECOGNIZE_TEST_IMAGE, "test_result")
