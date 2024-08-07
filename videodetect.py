import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('./yolov8n.pt')

# Load the video capture
video_path = './testvideo.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center_x = int(frame_width / 2)
frame_center_y = int(frame_height / 2)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID', 'MJPG', etc.
out = cv2.VideoWriter('resultvideo.mp4', fourcc, fps, (frame_width, frame_height))

frame_cnt = 0
# Loop through the frames of the video
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # cv2.circle(frame, (frame_center_x, frame_center_y), 5, (255, 0, 0), -1)

    if success:
        results = model.track(frame)

        for result in results:
            # Get the class names
            classes_names = result.names

            # Iterate over each box
            for box in result.boxes:
                # Check if confidence is greater than 40 percent
                if box.conf[0] > 0.4:
                    # Get coordinates
                    [x1, y1, x2, y2] = box.xyxy[0]

                    # Convert to int
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get the class
                    cls = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else -1
                    # Get the class name
                    class_name = classes_names[cls] if cls != -1 and cls < len(classes_names) else 'Unknown'

                    if class_name == 'Unknown':
                        # Calculate detect_center_x and detect_center_y
                        detect_center_x = int(((x2 - x1) / 2) + x1)
                        detect_center_y = int(((y2 - y1) / 2) + y1)

                        # Calculate x_center_gap and y_center_gap
                        x_center_gap = int(frame_center_x - detect_center_x)
                        y_center_gap = int(frame_center_y - detect_center_y)

                        cv2.arrowedLine(frame, (detect_center_x, detect_center_y), (frame_center_x, frame_center_y), (0, 255, 0), 3)
                        cv2.circle(frame, (detect_center_x, detect_center_y), 5, (255, 0, 0), -1)



                    # Get the track ID
                    track_id = box.id[0] if hasattr(box, 'id') and box.id is not None else None

                    # Draw the rectangle and track ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    label = f'{class_name} {box.conf[0]:.2f}'

                    # Draw track ID
                    if track_id is not None:
                        label += f' ID:{track_id}'
                    
                    # Draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

                    # Put the class name and confidence on the image
                    cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                    # Draw box height and width
                    box_height, box_width = int(x2 - x1), int(y2 - y1)
                    cv2.putText(frame, f'Height: {box_height}, Width: {box_width}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Write the frame into the file 'resultvideo.mp4'
        out.write(frame)
    else:
        break

# Release the video capture and writer objects
cap.release()
out.release()
