import cv2

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