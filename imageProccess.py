import cv2
import numpy as np
import matplotlib.pyplot as plt


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

# Draws lines on the zeroes_like of the image
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

    return bird_eye


def preprocess_image(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define yellow color range and create a mask
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

def detect_yellow_line(binary_warped):
    # Identify x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    return nonzerox, nonzeroy

def fit_polynomial(x, y):
    # Fit a second order polynomial
    try:
        fit = np.polyfit(y, x, 2)
    except:
        print("")
    return fit

# Determines the curve radius of the road
# If it's positive, the road bends right. Otherwise it bends towards the left
# Pass the bird's-eye image for optimal results
def calculate_curvature(image):
    binary_warped = preprocess_image(image)

    # Detect yellow line
    x, y = detect_yellow_line(binary_warped)

    # Fit polynomial to the yellow line
    fit = fit_polynomial(x, y)

    # Calculate curvature
    y_eval = binary_warped.shape[0] - 1

    # Calculate the radius of curvature
    curvature = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.abs(2 * fit[0])

    if fit[0] > 0:
        curvature = -curvature  # Left curve

    
    return curvature


# Determines the middle line's angle in the image
# Pass the bird's-eye image for optimal results
def calculate_angle(image):
    binary_warped = preprocess_image(image)

    # Detect yellow line
    x, y = detect_yellow_line(binary_warped)

    # Fit polynomial to the yellow line
    fit = fit_polynomial(x, y)

    # Calculate curvature
    y_eval = binary_warped.shape[0] - 1

    curvature = calculate_curvature(image)

    if abs(curvature) > 22000:
        return 0
    elif abs(curvature) > 12000:
        return 0.058 * curvature
    

    # Calculate the derivative of the polynomial
    derivative = 2 * fit[0] * y_eval + fit[1]
    # Calculate the angle in radians
    angle_rad = np.arctan(derivative)
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def steerFromCurvature(curvature):
    if abs(curvature) > 20000:
        return 0
    else:
        return 0.058 * curvature
    

# Make a log.md to use this. MD format is used to increase readability of logs with colors
def logCurvature(curvature):
    with open('log.md', 'a') as file:
        if curvature < 12000:
            file.write(f'<p style="color:red;">{curvature}</p>')    # Sharp bend: red
        elif 12000 <= curvature and curvature < 22000:
            file.write(f'<p style="color:blue;">{curvature}</p>' )    # Moderate bend: blue
        else:
            file.write(f'<p style="color:black;">{curvature}</p>')    # Straight road: black


def autoPassRightObstacle(image, car, time):
    frame, steering_deg = image_processor(image, 400)
    print(steering_deg)
    car.setSpeed(15)
    Steering -= 70
    print("Steering", Steering)
    car.setSteering(Steering)
    time.sleep(2)
    Steering += 140
    car.setSteering(Steering)
    time.sleep(2)
    Steering -= 70