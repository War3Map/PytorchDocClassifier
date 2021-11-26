import cv2
import numpy as np


def correct_rotation(image_path: str) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Corrects text rotation in image

    :param image_path: path to source image

    :return: result_image, rotation_angle, source_image
    """
    image = cv2.imread(image_path)
    print(type(image))
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    print(angle)
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    # TODO: correct rotation angle when near 90 degrees
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated, angle, image


if __name__ == '__main__':
    TEST_PATH = r"C:\Users\IVAN\Desktop\Texts\test\printed\doc21r.png"
    # TEST_PATH = r"C:\Users\IVAN\Desktop\Texts\test\printed\doc21r_serious.png"

    rotated, angle, source = correct_rotation(TEST_PATH)

    # draw the correction angle on the image so we can validate it
    cv2.putText(source, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imshow("Input", source)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
