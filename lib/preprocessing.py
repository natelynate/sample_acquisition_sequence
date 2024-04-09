import numpy as np
import cv2


def generate_landmark_image(landmarks, size):
    """Create binary array representing target facial landmarks"""
    base = np.zeros((size[0], size[1]))
    for landmark in landmarks:
        x, y = landmark
        base[y][x] = 1
    return base
    
