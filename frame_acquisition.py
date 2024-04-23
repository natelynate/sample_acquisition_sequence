"""
This is a sample acquisition sequence which captures webcam frames 
while performing either:
1. Sample calibration sequence where participant presses Enter key in response to a random dot location
2. Capture continuously from a livestream webcam for a desired duration

Set 'mode' parameter to 'calibration' or 'stream' to use respective utilities
default is 'calibration'
"""
from face_estimator import FaceEstimator
from lib.face import Face
from lib.eye import Eye
from lib.configs import *
import cv2
import numpy as np
import os
import winsound
from collections import deque


def main(mode="calibration", screen_width=1920, screen_height=1080):
    dot_radius = 4  # Radius of the dot
    instructions = "Press 'x' to start calibration sequence"  # User instructions
    instruction_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    base_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Display user instructions in a pop-up window
    cv2.namedWindow("Instructions", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Instructions", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.putText(instruction_image, instructions, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Instructions", instruction_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap = cv2.VideoCapture(0) # initialize webcam

    # Display current media input
    message = "Adjust your face to be at the center of the screen. Do not move until the calibration is over. Press Enter to continue"
    while True:
        _, frame = cap.read()
        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Current Media Input", frame)
        if cv2.waitKey(1) == 13:
            break

    cap = cv2.VideoCapture(0) # initialize webcam
    estimator = FaceEstimator()
    
    # Prepare a grid of gaze points
    gap = 10
    points = [(x, y) for x in range(0, screen_width, gap) for y in range(0, screen_height, gap)]
    queue = deque()
    np.random.shuffle(points)  # Shuffle the points
    
    # Append all gaze points to the queue 
    for point in points:
        queue.append(point)
        
    img_num = 0 # current order of the frame
    while queue:
        img_num += 1
        # Randomly generate dot coordinates
        dot_x, dot_y = queue.popleft() 
         # Draw a red dot on the blank image
        dot_color = (0, 0, 255)  # Full red color
        dot_image = cv2.circle(base_image, (dot_x, dot_y), dot_radius, dot_color, -1)

        # Show blank image with red dot fullscreen
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Calibration", dot_image)


        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # 13 is the ASCII code for Enter key
            _, frame = cap.read()
            face = estimator.inspect(frame) # detect face
            if face:
                face.fit() # fit face
                left_eye, right_eye = Eye(face.frame, face.landmarks[27:33]), Eye(face.frame, face.landmarks[27:33]) # Initialize eye objects
                # calculate average EAR
                avg_EAR = (left_eye.calc_EAR() + right_eye.calc_EAR()) / 2
                print(avg_EAR)
                if avg_EAR >= MINIMUM_EAR: # If eyes are open
                    winsound.Beep(1000, 500) # Notification
                    face.gaze_point = (dot_x, dot_y) # register label
                    face.save(str(img_num)) # save face object
                    left_eye.save(str(img_num) + "lefteye")
                    right_eye.save(str(img_num) + "righteye")
                else:
                    queue.append((dot_x, dot_y))
                    img_num -= 1
            else:
                queue.append((dot_x, dot_y))
                img_num -= 1
                    
        else:
            cap.release()
            cv2.destroyAllWindows()
            break
        base_image.fill(0)
        
if __name__ == "__main__":
    mode = "calibration"
    main(mode)
x