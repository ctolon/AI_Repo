import cv2
import pathlib
print("Python-openCV API Version:", cv2.__version__)

# Get Path infos
CURRENT_PATH = str(pathlib.Path().resolve())
IMAGE_PATH = CURRENT_PATH + '/images/m1.png'

# Read Image as Colored, gray and unchanged
image_color = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
image_grayscale = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
image_unchanged = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)

# Print Path
print("Image Path: ",IMAGE_PATH)

# Height - Width
h, w = image_color.shape[:2]
print("Height = {},  Width = {}".format(h, w))

# by passing in 100, 100 for height and width.
(B, G, R) = image_color[100, 100]

# Displaying the pixel values
print("R = {}, G = {}, B = {}".format(R, G, B))

# We can also pass the channel to extract 
# the value for a specific channel
B = image_color[100, 100, 0]
print("B = {}".format(B))

#Displays image inside a window (For GUI)
cv2.imshow(IMAGE_PATH, image_color)  
cv2.imshow(IMAGE_PATH, image_grayscale)
cv2.imshow(IMAGE_PATH, image_unchanged)

# For GUI
cv2.waitKey(0)
cv2.destroyAllWindows()