'''
@ 2023, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
''' 
import avisengine
import config
import time
import cv2
from imageProccess import *
from barrier import *



# Creating an instance of the Car class
car = avisengine.Car()

# Connecting to the server (Simulator)
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)
# Counter variable
counter = 0
Steering=0
debug_mode = False

# Sleep for 3 seconds to make sure that client connected to the simulator 
time.sleep(3)

try:
    while(True):
        # Counting the loops
        counter = counter + 1

        # Set the power of the engine the car to 20, Negative number for reverse move, Range [-100,100]
        car.setSpeed(20)

        # Set the Steering of the car -10 degree from center, results the car to steer to the left
       
        car.setSteering(Steering)
        
        # Set the angle between sensor rays to 45 degrees, Use this only if you want to set it from python client
        # Notice: Once it is set from the client, it cannot be changed using the GUI
        car.setSensorAngle(45) 

        # Get the data. Need to call it every time getting image and sensor data
        car.getData()

        # Start getting image and sensor data after 4 loops
        if(counter > 4):
            # Returns a list with three items which the 1st one is Left sensor data\
            # the 2nd one is the Middle Sensor data, and the 3rd is the Right one.
            sensors = car.getSensors() 
            left_sensor = sensors[0]
            middle_sensor = sensors[1]
            right_sensor = sensors[2]
           
            
            # Returns an opencv image type array. if you use PIL you need to invert the color channels.
            image = car.getImage()
            h, w, c = image.shape
            #print(h, w)
            # Cropping an image
            cropped_image = image[165:h-100, 100:w+50]
            
            # Display cropped image
            cv2.imshow("cropped", cropped_image)
            if(not image is None):
                barrier_pos = processImage(cropped_image)
                print(barrier_pos,"#######")
                # Returns an integer which is the real time car speed in KMH
                carSpeed = car.getSpeed()
                if(debug_mode):
                    (f"Speed : {carSpeed}") 
                    (f'Left : {str(sensors[0])} | Middle : {str(sensors[1])} | Right : {str(sensors[2])}')
                if(middle_sensor<1500):
                    frame, steering_deg =  image_processor(image,400)
                    if(barrier_pos>100):
                            car.setSpeed(15)
                            Steering-=90
                            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",Steering)
                            car.setSteering(Steering)
                    elif(barrier_pos<100 and barrier_pos!=0):
                        frame, steering_deg =  image_processor(image,100)
                        height, width = frame.shape[:2]
                        Steering+=90
                        car.setSteering(Steering) 
                        print(width-40//4, barrier_pos+50, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    else:
                         car.setSteering(steering_deg)
                elif(right_sensor<1500):
                            frame, steering_deg =  image_processor(image,100)
                            height, width = frame.shape[:2]
                            if(width-40/4<barrier_pos+50):
                                print(width-40//4, barrier_pos+50, "!!!!!")
                                car.setSpeed(15)
                                Steering-=90
                                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",Steering)
                                car.setSteering(Steering) 
                            else:
                                 car.setSteering(steering_deg) 
                elif(left_sensor<1500):
                            frame, steering_deg =  image_processor(image,100)
                            height, width = frame.shape[:2]
                            if(width-40/4>barrier_pos-50):
                                print(width-40//4, barrier_pos+50, "!!!!!")
                                car.setSpeed(15)
                                Steering+=90
                                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",Steering)
                                car.setSteering(Steering) 
                            else:
                                 car.setSteering(steering_deg) 
                elif(middle_sensor==1500):
                    if(barrier_pos==0): Steering=0
                    frame, steering_deg =  image_processor(image,100)
                    car.setSteering(steering_deg)
                #bird_eye_view(image)
                # Showing the opencv type image
                #cv2.imshow('frame',frame)
                if cv2.waitKey(10) == ord('q'):
                    break
            time.sleep(0.001)

finally:
    car.stop()
   
   