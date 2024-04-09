import dlib
import os
import cv2
from typing import List, Tuple
from lib.face import Face



class FaceEstimator():
    """Create estimator object for detecting face in a frame and landmark coordinates"""
    def __init__(self):
        model_path = "shape_predictor_68_face_landmarks.dat"
        self._face_detector = dlib.get_frontal_face_detector()
        self._landmark_predictor = dlib.shape_predictor(model_path) 
        self.face = None
    
    def inspect(self, frame) -> List:
        """Inspect if a face can be detected and return a Face object, else returns None"""
        face_candidates = self._face_detector(frame)
        if len(face_candidates) == 1:
            boundingbox = face_candidates[0] # returns dlib rectangle object
            landmarks = self._landmark_predictor(frame, boundingbox)

            # Convert dlib's object type into native python types. Into List[Tuple[int, int]]
            landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)]
            boundingbox = [(boundingbox.left(), boundingbox.top()), (boundingbox.right(), boundingbox.bottom())]
            
            return Face(frame, boundingbox, landmarks)
        else:
            return None
            
        

        
             


    

        


