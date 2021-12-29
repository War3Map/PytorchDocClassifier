import math
from pathlib import Path
from time import time
from pprint import pprint

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from recognizer.image_correction import correct_rotation
from recognizer.settings import TESSERACT_PATH, RECOGNIZE_TEST_IMAGE, TEST_DATA_PATH

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def show_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image


def prepare_text(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite("res.png", res)

    # Create horizontal kernel and dilate to connect text characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate = cv2.dilate(mask, kernel, iterations=1)

    # Find contours and filter using aspect ratio
    # Remove non-text contours by filling in the contour
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if ar < 5:
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

    # Bitwise dilated image with mask, invert, then OCR
    result = 255 - cv2.bitwise_and(dilate, mask)

    cv2.imwrite("masked.png", dilate)
    cv2.imwrite("dilated.png", mask)
    cv2.imwrite("result.png", result)

    return result


def tesseract_correct_angle(image):
    """
    Rotates upside down or 90 degrees image
    :param image: image data
    :return: rotated image
    """
    # detect text rotation angle
    # start_time = time()
    info = pytesseract.image_to_osd(image, output_type=Output.DICT)
    print(info)
    angle = info["orientation"]
    # end_time = time() - start_time
    # print(f"Определение угла({angle}): {end_time}")

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
        print("Поворот:!")
    return image


def tesseract_recognize_text(image, precise: bool = False):
    """
    Returns text from the image
    :param precise:
    :param image:
    :return:
    """
    # Adding custom options
    # Page segmentation mode
    # 6 = Assume a single uniform block of text.
    # OCR Engine modes:
    # 3 = Default, based on what is available.

    start_time = time()
    custom_config = r'--oem 3 --psm 6'

    results = pytesseract.image_to_data(image,
                                        output_type=Output.DICT,
                                        config=custom_config,
                                        lang='rus')

    # result = pytesseract.image_to_string(image,
    #                                      config=custom_config,
    #                                      lang='rus')

    pprint(results, compact=True)

    for i in range(0, len(results["text"])):
        # We can then extract the bounding box coordinates
        # of the text region from  the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]

        # We will also extract the OCR text itself along
        # with the confidence of the text localization
        text = results["text"][i]
        conf = int(results["conf"][i])
        level = int(results["level"][i])
        # image = cv2.rectangle(image,
        #                       (x, y),
        #                       (x + w, y + h),
        #                       (255, 0, 0), 2)
        confidiencies = []
        confidiencies.append(-1)
        confidiencies.extend(range(2, 12))
        if conf in confidiencies:
            image = cv2.rectangle(image,
                                  (x, y),
                                  (x + w, y + h),
                                  (255, 0, 0), 2)

        # filter out weak confidence text localizations
        # if conf > args["min_conf"]:
        #     # We will display the confidence and text to
        #     # our terminal
        #     print("Confidence: {}".format(conf))
        #     print("Text: {}".format(text))
        #     print("")
        #
        #     # We then strip out non-ASCII text so we can
        #     # draw the text on the image We will be using
        #     # OpenCV, then draw a bounding box around the
        #     # text along with the text itself
        #     text = "".join(text).strip()
        #     cv2.rectangle(images,
        #                   (x, y),
        #                   (x + w, y + h),
        #                   (0, 0, 255), 2)
        #     cv2.putText(images,
        #                 text,
        #                 (x, y - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1.2, (0, 255, 255), 3)

    cv2.imwrite("Test.png", image)
    end_time = time() - start_time
    print(f"Распознавание: {end_time}")


def get_text_bb(image):
    """
    Get bounding boxes of text inside image
    :param image:
    :return:
    """
    start_time = time()
    # run tesseract, returning the bounding boxes
    # data = pytesseract.image_to_data(image, lang="rus")
    # print(data)

    # boxes = pytesseract.image_to_boxes(image)  # also include any config options you use
    # # boxrs = pytesseract.get_boxes
    # print("Get boxes!")
    # # draw the bounding boxes on the image
    # x_left_top, y_left_top, x_bot_right, y_bot_right = w, h, 0, 0
    # max_dist = 0
    # tops, bots = [], []
    # for b in boxes.splitlines():
    #     b = b.split(' ')
    #     image = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    #     x1, y1, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
    #     tops.append((x1, y1))
    #     bots.append((x2, y2))
    #     if x1 < x_left_top and y1 < y_left_top:
    #         x_left_top, y_left_top = x1, y1
    #
    #     # if x2 >= x_bot_right and y2 > y_bot_right:
    #     if math.sqrt(x2 ** 2 + y2 ** 2) > max_dist:
    #         x_bot_right, y_bot_right = x2, y2
    #
    # print(min(tops), max(bots))
    #
    # print((x_left_top, y_left_top), (x_bot_right, y_bot_right))
    #
    # xm, ym = min(tops)
    # xmx, ymx = max(bots)
    #
    # image = cv2.circle(image, (xm, ym), 8, (255, 0, 255), 5)
    # image = cv2.circle(image, (xmx, ymx), 5, (0, 255, 255), 5)
    #
    # image = cv2.circle(image, (x_left_top, y_left_top), 8, (255, 0, 0), 5)
    # image = cv2.circle(image, (x_bot_right, y_bot_right), 5, (255, 255, 0), 5)
    # # image = cv2.rectangle(image, (x_left_top, y_left_top), (x_bot_right, y_bot_right), (0, 255, 0), 2)
    # image = cv2.rectangle(image, (xm, ym), (xmx, ymx), (0, 255, 0), 2)
    # image = cv2.rectangle(image, (x_left_top, y_left_top), (x_bot_right, y_bot_right), (0, 255, 255), 2)

    # for (xt, yt), (xb, yb) in zip(tops, bots):
    #     image = cv2.rectangle(image, (xt, yt), (xb, yb), (0, 255, 0), 2)
    #     print((xt, yt), (xb, yb))

    # d = pytesseract.image_to_data(image, output_type=Output.DICT)
    # n_boxes = len(d['level'])
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    end_time = time() - start_time
    print(f"Time: {end_time}")


def tesseract_recognize(image_path):
    """

    :param image_path:
    :return:
    """

    image = cv2.imread(image_path)
    h, w, _ = image.shape  # assumes color image

    # image = prepare_text(image)
    # show_image(image)

    # show_image(image)
    # show annotated image and wait for keypress

    cv2.imwrite("corrected.png", image)

    # cv2.imshow("Test", image)
    # cv2.waitKey(0)

    # correct rotation, align text
    # start_time = time()
    # image, _, _ = correct_rotation(image)
    # end_time = time() - start_time
    # print(f"Корректировка угла: {end_time}")
    # show_image(image)

    image = tesseract_correct_angle(image)
    result_text = tesseract_recognize_text(image)

    return result_text


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
    test_result_path = Path(TEST_DATA_PATH).resolve() / "result.txt"
    recognize_image(RECOGNIZE_TEST_IMAGE, str(test_result_path))
