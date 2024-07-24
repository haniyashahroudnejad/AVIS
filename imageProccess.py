import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gaussian, 50, 150, apertureSize=3)
    return canny


def croppImg(image, mask):
    return cv2.bitwise_and(image, mask)


def maskImage(image):
    height = image.shape[0]
    triangle = np.array([
        [(45, 395), (height, 395), (265, 150)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    maskImage = croppImg(image, mask)
    return maskImage


def displayLines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

    return line_image


def houghLines(canny):
    lines = cv2.HoughLinesP(maskImage(canny), 1, np.pi/180,
                            100, np.array([]), minLineLength=40, maxLineGap=5)
    return lines


def position_detect(detected_line):
    POSITION = np.mean(np.where(detected_line > 0), axis=1)
    return POSITION


def error_detect(POSITION, avg):
    error = avg - POSITION[1]
    print("error=", error)
    return error


def calculate_steering(error):
    # Error < 0 --> turn right
    # Error > 0 --> turn left
    if (error > 90):
        change = 1
    elif (error < -50):
        change = 1.2
    else:
        change = 0.7
    steering_degree = -1 * error * change
    return steering_degree


def image_processor(image, avg):
    cann = canny(image)
    line_image = displayLines(image, houghLines(cann))
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    up = np.array([129, 255, 255])
    down = np.array([103, 200, 200])
    line_selection = cv2.inRange(combo_image, down, up)
    POSITION = position_detect(line_selection)
    error = error_detect(POSITION, avg)
    steering_degree = calculate_steering(error)
    return combo_image, steering_degree


def colorMask(image, high, low):
    # img to hsv img
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # find yellow color
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    return mask


def image_processed_middle_line_getter(image):
    yellowMask = colorMask(image, [45, 255, 255], [22, 93, 0])
    line_image = houghLines(yellowMask)
    return line_image


def bird_eye_view(frame):
    height = 550
    width = 550  # 512
    # Define source points (corners of the trapezoid)
    src = np.float32([
        [20, 260],
        [480, 260],
        [480, 350],
        [20, 350]
    ])
    # Define destination points (corners of the rectangle)
    dst = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the image to get the bird's eye view
    bird_eye = cv2.warpPerspective(frame, M, (width, height))
    cv2.imshow("bird_eye", bird_eye)
