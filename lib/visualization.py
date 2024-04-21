import cv2
import numpy as np

def visualize(frame, points=None, dot_size=2):
    # Define properties
    dot_color = (0, 255, 0)  # Green color in BGR format
    
    
    # Make a copy of the image to draw dots on
    dot_image = frame.copy()
    
    # Iterate through each landmark point
    if points:
        for landmark in points:  # Assuming 'points' is the list of landmarks
            # Draw each point on the image
            dot_image = cv2.circle(dot_image, (landmark[0], landmark[1]), dot_size, dot_color, -1)
        
    # Display the image with landmarks
    cv2.imshow("Labeled Image", dot_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_npy(file_directory):
    # Load the NumPy array from file
    frame = np.load(file_directory)

    # Check if conversion is needed to uint8
    if frame.dtype != np.uint8:
        # Normalize and convert (assuming the original range is 0 to 1 for floats)
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    return frame
