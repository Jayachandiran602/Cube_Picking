import numpy as np
import cv2

def display_image(image, name, scale=2, wait=False):
    """ 
    function to display an image 
    :param image: ndarray, the image to display
    :param name: string, a name for the window
    :param scale: int, optional, scaling factor for the image
    :param wait: bool, optional, if True, will wait for click/button to close window
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, image.shape[1]*scale, image.shape[0]*scale)
    cv2.imshow(name, image)
    cv2.waitKey(0 if wait else 1)
    
def detect_cubes(image):
    """
    Detect cubes in the image.
    :param image: ndarray, the image from the camera
    :return: list of tuples (x, y, w, h) for detected cubes
    """
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the color range for detecting the cubes
    lower_color = np.array([20, 100, 100])
    upper_color = np.array([30, 255, 255])
    
    # Create a mask for the color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cubes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cubes.append((x, y, w, h))
    
    return cubes
