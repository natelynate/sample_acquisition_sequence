# Sample acquisition sequence

import cv2
import numpy as np
import os
import time
import json
from detect_landmarks import detect_landmark
from gaze_tracking import GazeTracking
import pandas as pd

def save_image(image, folder, filename:str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), image)


def save_dot_loc(x, y, folder, filename:str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, filename + '.txt'), 'w') as f:
        f.writelines(f'x:{x} y:{y}')


def save_json(coords:dict, filename:str):
    # Specify the directory path
    json_dir = os.path.join(BASE_URL, 'json')
    
    # Set json file name
    json_file_path = os.path.join(json_dir, f'{filename}_landmarks.json')
    
    # Open the file for writing
    with open(json_file_path, 'w') as json_log:
        # Dump the dictionary to the JSON file
        json.dump(coords, json_log)
        print(f"JSON file saved as: {json_file_path}")


def apply_alignment(landmarks, target_landmarks):
    pass


def main(source='', screen_width=1920, screen_height=1080):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    gaze = GazeTracking()
    frames = 0
    counted_frames = 0
    collected_data = pd.DataFrame()

    # Configuration
    screen_width = 1920  # Example screen width
    dot_radius = 4  # Radius of the dot
    instructions = "Press 'x' to start calibration sequence"  # User instructions

    # Create a blank image
    instruction_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    blank_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    # Display user instructions in a pop-up window
    cv2.namedWindow("Instructions", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Instructions", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.putText(instruction_image, instructions, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Instructions", instruction_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    while True:
        frames += 1

        # Randomly generate dot coordinates
        dot_x = np.random.randint(0, screen_width)
        dot_y = np.random.randint(0, screen_height)

        # Draw a red dot on the blank image
        dot_color = (0, 0, 255)  # Full red color
        dot_image = cv2.circle(blank_image, (dot_x, dot_y), dot_radius, dot_color, -1)

        # Show blank image with red dot fullscreen
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Calibration", dot_image)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # 13 is the ASCII code for Enter key
            # Capture image from webcam
            ret, frame = cap.read()
            if ret:
                gaze.refresh(frame)
                image_name = f'{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                if gaze.pupils_located:
                    # Save captured image along with related data
                    save_image(gaze.annotated_frame(), 'images', image_name)
                    # save_dot_loc(dot_x, dot_y, 'coordinates', image_name)
                    # save_json(gaze.landmark_coordinates, image_name)
                    counted_frames += 1
                    new_row = gaze.landmark_coordinates
                    new_row['name'] = image_name
                    new_row['dot_x'] = dot_x
                    new_row['dot_y'] = dot_y 
                    collected_data = pd.concat([collected_data, pd.DataFrame(new_row, index=[0])])
                else:
                    save_image(frame, 'images', image_name)
                    
        blank_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break

    # Release webcam
    cap.release()
    cv2.destroyAllWindows()

    print(f"{counted_frames} positive frames /{frames} frames")
    collected_data.reset_index()
    collected_data.to_csv('./collected_data.csv')

if __name__ == "__main__":
    BASE_URL = os.getcwd()
    main()
