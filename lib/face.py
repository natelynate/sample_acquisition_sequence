"""
Class definition for face object. Face Object contains these as attributes:
1. BoundingBox coordinates(Topleft, BottomRight) of a detected face region
2. Discovered 3D coordinates of facial landmarks as per stated target points of interest
"""
import cv2
import os
import numpy as np
import json
from lib.preprocessing import generate_landmark_image
from typing import Tuple, List
from lib.configs import LANDMARK_INDICES_OF_INTEREST

# These are indices of facial landmarks of interest that will be used for modeling

class Face():
    def __init__(self, frame, boundingbox, landmarks, gaze_point=None):
        self._LANDMARK_INDICES_OF_INTEREST = LANDMARK_INDICES_OF_INTEREST
        self.frame = frame # original frame
        self.boundingbox = boundingbox # bb around face region on original frame. List[Tuple[int, int], Tuple[int, int]]
        self.landmarks = [point for idx, point in enumerate(landmarks) if idx in self._LANDMARK_INDICES_OF_INTEREST]
        self.gaze_point = gaze_point # 2D Gaze target onscreen location 

    def _prep_save(self, base):
        frame_dir = os.path.join(base, 'frame')
        lm_img_dir = os.path.join(base, 'lm_image')
        lm_dir = os.path.join(base, 'landmark')
        label_dir = os.path.join(base, 'labels')

        # Confirm directory existence, else create new 
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(lm_img_dir, exist_ok=True)
        os.makedirs(lm_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        return frame_dir, lm_img_dir, lm_dir, label_dir

    def save(self, filename, base='./data/face/', normalize=True):
        """Save current attributes at specified save directories. It automatically create subfolders 
        (frame, boundingbox, landmark) if subfolder names are not found in the base argument
        if msg argument is set to True, success/failure message will be displayed at the end. 
        """
        frame_dir, lm_img_dir, lm_dir, label_dir = self._prep_save(base)
        np.save(os.path.join(frame_dir, f'{filename}'), self.frame / 255.0)
        np.save(os.path.join(lm_img_dir, f'{filename}'), generate_landmark_image(self.landmarks, (self.frame.shape[0], self.frame.shape[1])))

        # Write metainfo to the respective directories
        with open(os.path.join(lm_dir, f'{filename}_lm.json'), 'w') as file: # 2d landmark coordinates
            json.dump(self.landmarks, file)

        with open(os.path.join(label_dir, f'{filename}_label.json'), 'w') as file:
            json.dump(self.gaze_point, file)


    def fit(self, size=(244, 244), padding=10):
        """Applies preprocessing steps to the current Face object. Preprocessing steps include cropping and resizing.
           landmark coordinates are aligned according to the undergoing transformation."""
        try:
            # Perform Cropping 
            cropped = self._crop(padding=padding) 
            
            newlandmarks = self._align(dx=self.boundingbox[0][0]-padding, 
                                       dy=self.boundingbox[0][1]-padding) # align
            
            self._refresh(cropped, newlandmarks) # apply changes
            
            # Perform Resizing
            cropped_and_resized = self._resize(size=size)
            x, y, _ = self.frame.shape 
            
            dx, dy = size[0] / x, size[1] / y
            newlandmarks = self._align(dx, dy)
            self._refresh(cropped_and_resized, newlandmarks)
            return True
        except:
            raise ValueError("Error during fit process")

        
    def _align(self, dx, dy):
        """Realign landmark points depending on the transformation. 
           If dx & dy are integers specifying the top left coord of the boundingbox, points are aligned relative to the new top-left corner. 
           If dx & dy are floats specifying the resizing ratios, points are algined accordingly. 
        """
        aligned_landmarks = [] # List for containing newly aligned coords
        
        if type(dx) == int and type(dy) == int: # Realign in conjunction to cropping
            for idx in range(len(self.landmarks)):
                point = self.landmarks[idx] 
                aligned_landmarks.append((point[0]-dx, point[1]-dy))

        elif type(dx) == float and type(dy) == float: # Realign in conjunction to resizing 
            for idx in range(len(self.landmarks)):
                coord_x, coord_y = self.landmarks[idx]
                n_coord_x = self._ceil(int(coord_x * dx), 244-1) 
                n_coord_y = self._ceil(int(coord_y * dy), 244-1)
                aligned_landmarks.append((n_coord_x, n_coord_y))
        return aligned_landmarks


    def _crop(self, padding):
        """Perform cropping around the bounding box with padding"""
        try:
            left, top = self.boundingbox[0]
            right, bottom = self.boundingbox[1]
            return self.frame[top-padding:bottom+padding, left-padding:right+padding] # Apply padding
        except:
            raise ValueError("Error during cropping")


    def _refresh(self, frame, landmarks):
        self.frame = frame
        self.landmarks = landmarks


    def _resize(self, size):
        return cv2.resize(self.frame, size)
    

    def _ceil(self, n, threshold):
        if n > threshold:
            n = threshold
        return n

