import cv2
import numpy as np



def canny(image):
    gray = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
    gaussian = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(gaussian,50,150,apertureSize=3)
    return canny

def croppImg(image , mask):
    return cv2.bitwise_and(image,mask)

def maskImage(image):
    height = image.shape[0]
    triangle = np.array([
	[(45,395),(height,395),(265,150)]
	])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,triangle,255)
    maskImage = croppImg (image, mask)
    cv2.imshow('image',maskImage)
    return maskImage   

def displayLines(image, lines):
    line_image = np.zeros_like(image)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
           
    return line_image

    
def houghLines(canny):
    lines=cv2.HoughLinesP(maskImage(canny),1,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    return lines
   
def position_detect(detected_line):
	POSITION = np.mean(np.where(detected_line > 0),axis = 1)
	return POSITION


def error_detect(POSITION,avg):
    error = avg - POSITION[1]
    print("error=",error)
    return error


bfError= 0

def calculate_steering(error):
	# Error < 0 --> turn right
	# Error > 0 --> turn left
    global bfError 
    if(bfError<30 and bfError>0 and error<0 and (bfError-error)>=12):
         print("thissssssssssssssss")
         print("bfError=",bfError)
         error+=40
         change = 1.5
    elif(error>48) :
        print("1")
        error+=20
        change = 0.85
    elif ( error<-40 and error>-70 ) : 
        print("2")
        error+=-error+30
        change = 0.8
    elif( error<-41) :
        print("3")
        error+=40
        change = 0.4
    else : change=0.4
    steering_degree = -1 * error * change
    bfError = error
    return steering_degree

def yellowLine_detect(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of the yellow regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the yellow line
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract y-coordinates of contour points
        y_coords = largest_contour[:, 0, 1]
  
        y_coords = np.mean(y_coords)/2
        return y_coords
    

def image_processor(image,x):
    x= -70
    y_coords= yellowLine_detect(image)+x

    cann = canny(image) 
    line_image = displayLines(image,houghLines(cann))
    combo_image = cv2.addWeighted(image,0.8,line_image,1,1)
    up = np.array([129,255,255])
    down = np.array([103, 200, 200])
    line_selection = cv2.inRange(combo_image, down, up)
    POSITION = position_detect(line_selection)
    error = error_detect(POSITION,y_coords)
    steering_degree = calculate_steering(error)
    return combo_image,steering_degree      

def colorMask(image , high , low):
    #img to hsv img
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    #find yellow color
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    return mask    
        
def image_processed_middle_line_getter(image):
    yellowMask = colorMask(image , [45, 255, 255] , [22, 93, 0] )
    line_image = houghLines(yellowMask)
    return line_image

def bird_eye_view(frame):
    height=512
    width = 512
    # Define source points (corners of the trapezoid)
    src = np.float32([
        [70, 260],
        [420, 260],
        [420, 350],
        [70, 350]
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
    cv2.imshow("bird_eye",bird_eye)
