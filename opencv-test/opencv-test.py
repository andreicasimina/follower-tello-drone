import os
import cv2

directory = os.path.dirname(os.path.abspath(__file__))

image_name = 'test.jpg'
image_file = os.path.join(directory, image_name)

cv2_image = cv2.imread(image_file)

# Show the image in a window
cv2.imshow('image_window', cv2_image)

# Wait for a key press (0 means wait indefinitely)
cv2.waitKey(0)

# Close all open windows
cv2.destroyAllWindows()