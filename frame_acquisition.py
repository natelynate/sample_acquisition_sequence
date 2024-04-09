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
import cv2
import numpy as np
import os
import winsound


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

    estimator = FaceEstimator()
    img_num = 0
    while True:
        img_num += 1
        # Randomly generate dot coordinates
        dot_x = np.random.randint(0, screen_width)
        dot_y = np.random.randint(0, screen_height)

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
            ret, frame = cap.read()
            if ret: 
                face = estimator.inspect(frame)
                if face:
                    winsound.Beep(1000, 500) # Notification
                    face.fit()
                    face.gaze_point = (dot_x, dot_y) # register label
                    face.save(str(img_num)) # save the data
        else:
            cap.release()
            cv2.destroyAllWindows()
            break
        base_image.fill(0)
        
if __name__ == "__main__":
    mode = "calibration"
    main(mode)
