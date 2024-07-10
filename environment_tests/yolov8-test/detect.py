import os, cv2
from ultralytics import YOLO

directory = os.path.dirname(os.path.abspath(__file__))

# Load the model
model = YOLO(os.path.join(directory, 'yolov8n.pt'))

# Load the video capture
image_name = 'test.jpg'
image_file = os.path.join(directory, image_name)

cv2_image = cv2.imread(image_file)

results = model.predict(cv2_image)

for result in results:
    # get the classes names
    classes_names = result.names

    # iterate over each box
    for box in result.boxes:
        # check if confidence is greater than 40 percent
        if box.conf[0] > 0.4:
            # get coordinates
            [x1, y1, x2, y2] = box.xyxy[0]
            # convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # get the class
            cls = int(box.cls[0])

            # get the class name
            class_name = classes_names[cls]

            # draw the rectangle
            cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # put the class name and confidence on the image
            cv2.putText(cv2_image, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
# show the image
cv2.imshow('cv2_image', cv2_image)

cv2.waitKey(0)

cv2.destroyAllWindows()