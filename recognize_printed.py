import cv2
import pytesseract

TESSERACT_PATH = r"D:\Tesseract\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def tesseract_recognize(image_path):
    """

    :param image_path:
    :return:
    """
    img = cv2.imread(image_path)

    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_string(img, config=custom_config, lang='rus')
    return result
