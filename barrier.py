import cv2
import numpy as np


def empty(a):
        pass


def stackImages(scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1] if rowsAvailable else imgArray[0].shape[1]
        height = imgArray[0][0].shape[0] if rowsAvailable else imgArray[0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2:
                        imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)

            
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
                if len(imgArray[x].shape) == 2:
                    imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    avg_x=0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # reduce noise by using area
        if area > 3000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, peri * 0.02, True)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            #print(aspect_ratio)
            # Assuming barriers have a specific aspect ratio range
            if 1 < aspect_ratio < 5.5:  # Adjust aspect ratio range as needed
                cv2.drawContours(imgContour, [cnt], -1, (255, 0, 255), 2)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(imgContour, "points: " + str(len(approx)), (x, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(imgContour, "area: " + str(int(area)), (x, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                # Print and store the coordinates of the points
                sum_x = 0.0
                sum_y = 0.0
                num = 0
                for point in approx:
                    point_coordinates = tuple(point[0])
                    sum_x += point_coordinates[0]
                    sum_y += point_coordinates[1]
                    num += 1
                    print("Point coordinates:", point_coordinates)
                    cv2.circle(imgContour, point_coordinates, 5, (0, 100, 0), cv2.FILLED)

                avg_x = sum_x / num
                avg_y = sum_y / num
                print("Average coordinates:", (avg_x, avg_y))
                if(avg_x>100):
                    print("right********")
                else:
                     print("left&&&&&&&&&&")
    return avg_x


def processImage(img):
            image = img
            cv2.namedWindow("parameters")
            cv2.resizeWindow("parameters", 640, 240)
            cv2.createTrackbar("Threshold1", "parameters", 40, 255, empty)
            cv2.createTrackbar("Threshold2", "parameters", 75, 255, empty)
        
            imgContour = image.copy()
            imgBlur = cv2.GaussianBlur(image, (7, 7), 1)
            imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)

            # Define the gray color range in HSV (adjust these values)
            lower_gray = np.array([96, 5, 96])
            upper_gray = np.array([110, 21, 206])
            mask = cv2.inRange(imgHSV, lower_gray, upper_gray)

            threshold1 = cv2.getTrackbarPos("Threshold1", "parameters")
            threshold2 = cv2.getTrackbarPos("Threshold2", "parameters")

            imgCanny = cv2.Canny(mask, threshold1, threshold2)
            kernel = np.ones((5, 5))
            imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
            avg = getContours(imgDil, imgContour)
            imgStack = stackImages(0.8, [[imgContour]])  # Modified to pass a 2D list
            cv2.imshow("result", imgStack)
            return avg