import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('./head-detection.pt')

# Load the video capture
image_file = './test.jpg'

cv2_image = cv2.imread(image_file)

results = model.track(cv2_image)

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
            cls = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else -1
            # get the class name
            class_name = classes_names[cls] if cls != -1 and cls < len(classes_names) else 'Unknown'

            # get the track ID
            track_id = box.id[0] if hasattr(box, 'id') and box.id is not None else None

            # draw the rectangle and track ID
            cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (0, 0, 0), 2)
            label = f'{class_name} {box.conf[0]:.2f}'

            # draw track id
            if track_id is not None:
                label += f' ID:{track_id}'
            
            # draw the rectangle
            cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # put the class name and confidence on the image
            cv2.putText(cv2_image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
# show the image
cv2.imshow('cv2_image', cv2_image)

# save the image
cv2.imwrite('./result.jpg', cv2_image)

cv2.waitKey(0)

cv2.destroyAllWindows()