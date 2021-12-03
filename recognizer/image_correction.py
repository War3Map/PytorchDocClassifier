import cv2
import numpy as np

from recognizer.settings import CORRECTION_TEST_PATH


def correct_rotation(source_image) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Corrects text rotation in image

    :param source_image: image to process
    # :param image_path: path to source image

    :return: result_image, rotation_angle, source_image
    """
    # image = cv2.imread(image_path)
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    current_angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if current_angle < -45:
        current_angle = -(90 + current_angle)
    elif 75 < abs(current_angle) < 90:
        current_angle = (90 - current_angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        current_angle = -current_angle

    # rotate the image to deskew it
    h, w = source_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, current_angle, 1.0)
    rotated_image = cv2.warpAffine(source_image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

    return rotated_image, current_angle, source_image


if __name__ == '__main__':
    image = cv2.imread(CORRECTION_TEST_PATH)
    rotated, angle, source = correct_rotation(image)

    # draw the correction angle on the image so we can validate it
    cv2.putText(source, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    print("[INFO] rotation angle: {:.3f}".format(angle))
    cv2.imshow("Input", source)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
