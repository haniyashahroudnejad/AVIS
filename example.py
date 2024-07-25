'''
@ 2023, Copyright AVIS Engine
- An Example Compatible with AVISEngine version 2.0.1 / 1.2.4 (ACL Branch) or higher
'''
import avisengine
import config
import time
import cv2
from imageProccess import *
import matplotlib.pyplot as plt


# Creating an instance of the Car class
car = avisengine.Car()

# Connecting to the server (Simulator)
car.connect(config.SIMULATOR_IP, config.SIMULATOR_PORT)

# Counter variable
counter = 0
Steering = 0
debug_mode = False

# Sleep for 3 seconds to make sure that client connected to the simulator
time.sleep(3)

try:
    while (True):
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
        if (counter > 4):
            # Returns a list with three items which the 1st one is Left sensor data\
            # the 2nd one is the Middle Sensor data, and the 3rd is the Right one.
            sensors = car.getSensors()
            left_sensor = sensors[0]
            middle_sensor = sensors[1]
            right_sensor = sensors[2]

            # Returns an opencv image type array. if you use PIL you need to invert the color channels.
            image = car.getImage()
            if (not image is None):

                # Returns an integer which is the real time car speed in KMH
                carSpeed = car.getSpeed()
                if (debug_mode):
                    (f"Speed : {carSpeed}")
                    (f'Left : {str(sensors[0])} | Middle : {str(sensors[1])} | Right : {str(sensors[2])}')

                if ((right_sensor < 600) or middle_sensor < 1000):
                    frame, steering_deg = image_processor(image, 400)
                    car.setSpeed(15)
                    Steering -= 70
                    print("Steering", Steering)
                    car.setSteering(Steering)
                elif ((left_sensor < 600) or middle_sensor < 1000):
                    frame, steering_deg = image_processor(image, 113)
                    car.setSteering(290)
                else:
                    Steering = 0
                    frame, steering_deg = image_processor(image, 113)
                    car.setSteering(steering_deg)

                # Showing bird's-eye view:
                image = bird_eye_view(frame)
                
                curvature = calculate_curvature(image)
                # print(f'Radius of Curvature: {curvature:.2f} meters')

                # Log curvature in log.md
                with open('log.md', 'a') as file:
                    file.write(isRoadCurved(curvature))




                # # Plot the results
                # ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
                # fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

                # plt.imshow(binary_warped, cmap='gray')
                # plt.plot(fitx, ploty, color='yellow')
                # plt.xlim(0, binary_warped.shape[1])
                # plt.ylim(binary_warped.shape[0], 0)
                # plt.show()


                # ----------------------------------------------------------------------------



                # Showing the opencv type image
                cv2.imshow('frame', frame)
                if cv2.waitKey(10) == ord('q'):
                    break
            time.sleep(0.001)

finally:
    car.stop()
