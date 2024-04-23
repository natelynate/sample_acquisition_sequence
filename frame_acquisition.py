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
from datetime import datetime



def main(applicant_name, screen_width=1920, screen_height=1080):
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
    estimator = FaceEstimator()

    # Display current media input
    message = "Adjust your face to be at the center of the screen. Do not move until the calibration is over. Press Enter to continue"
    while True:
        _, frame = cap.read()
        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        face = estimator.inspect(frame) # detect face
        if face:
            cv2.rectangle(frame, face.boundingbox[0], face.boundingbox[1], (0, 255, 0), 2)
        
        cv2.imshow("Current Media Input", frame)
        if cv2.waitKey(1) == 13:
            break

    
    # Prepare a grid of gaze points
    gap = 10
    points = [(x, y) for x in range(0, screen_width, gap) for y in range(0, screen_height, gap)]
    queue = deque()
    np.random.shuffle(points)  # Shuffle the points
    
    # Append all gaze points to the queue 
    for point in points:
        queue.append(point)
    
    instance_num = 0
    while queue:
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
            face.gazepoint = (dot_x, dot_y)
            if face:
                face.fit(crop_eye=True) # fit face
                # calculate average EAR
                avg_EAR = (face.eyes[0].EAR + face.eyes[1].EAR) / 2
                if avg_EAR >= MINIMUM_EAR: # If eyes are open
                    instance_num += 1
                    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = '_'.join([current_time, applicant_name, str(instance_num)])
                    face.save(filename) # save face object
                    winsound.Beep(1000, 500) # Notification
                    print(filename)
                else:
                    queue.append((dot_x, dot_y))
                    
            else:
                queue.append((dot_x, dot_y))
        else:
            cap.release()
            cv2.destroyAllWindows()
            break
        base_image.fill(0)
        
if __name__ == "__main__":
    applicant_name = input("Type in the applicant's initial: ")
    main(applicant_name)