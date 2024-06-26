import numpy as np
import json
import os
from datetime import datetime

class Eye():
    """
    note: The frame required during initialization is a whole face not a cropped eye image
    """
    def __init__(self, frame, landmarks):
        self.landmarks = landmarks
        self.frame = frame
        self._prep()
       
    def _prep(self):
        self.boundingbox = self._set_boundingbox()
        self.frame = self._crop_out_eye(self.boundingbox, padding=5)
        self.EAR = self._calc_EAR()
        
    def _calc_EAR(self):
        """"""
        p1, p4 = self.landmarks[0][0], self.landmarks[3][0]
        p2, p6 = self.landmarks[1][1], self.landmarks[5][1]
        p3, p5 = self.landmarks[2][1], self.landmarks[4][1]
        return ((abs(p2-p6) + abs(p3-p5)) / (2 * abs(p1-p4))) 
         
    def _set_boundingbox(self):
        """Set boundingbox around the eye for future cropping actions"""
        min_x, max_x, min_y, max_y = self.frame.shape[1], 0, self.frame.shape[0], 0
        for lm in self.landmarks:
            min_x = min(lm[0], min_x)
            max_x = max(lm[0], max_x)
            min_y = min(lm[1], min_y)
            max_y = max(lm[1], max_y)
        return min_x, max_x, min_y, max_y
    
    def _crop_out_eye(self, boundingbox, padding=5):
        """Crop frames based on the given landmarks"""
        min_x, max_x, min_y, max_y = boundingbox
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
    
        # Assert that post-padded coord is within the asserted image sizes
        min_x = max(min_x - padding, 0)
        max_x = min(max_x + padding, self.frame.shape[1] - 1)
        min_y = max(min_y - padding, 0)
        max_y = min(max_y + padding, self.frame.shape[0] - 1)

        return self.frame[min_y:max_y+1, min_x:max_x+1]

    

