from ultralytics import YOLO
import cv2
import socket
import threading
import time

# Image recognition part
model = YOLO('./head-detection.pt', task='track')  # Initialize YOLO model with tracking
model.to('cuda')

# Function for receiving data
def udp_receiver(sock):
    while True:
        try:
            response, _ = sock.recvfrom(1024)
            print("Received data:", response.decode('utf-8'))
        except socket.timeout:
            print("Socket timeout, retrying...")
            continue
        except Exception as e:
            print("Error in udp_receiver:", e)
            break

# Function to initialize and return a socket
def initialize_socket(TELLO_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', TELLO_PORT))
    sock.settimeout(30.0)
    return sock

def command_thread(sock, TELLO_ADDRESS, message):
    sock.sendto(f'{message}'.encode('utf-8'), TELLO_ADDRESS)

def send_command(sock, TELLO_ADDRESS, message):
    thread = threading.Thread(target=command_thread, args=(sock, TELLO_ADDRESS, message, ))
    thread.daemon = True
    thread.start()

def takeoff(sock, TELLO_ADDRESS):
    send_command(sock, TELLO_ADDRESS, 'command')
    time.sleep(1)
    send_command(sock, TELLO_ADDRESS, 'streamon')
    time.sleep(1)
    send_command(sock, TELLO_ADDRESS, 'takeoff')
    time.sleep(5)
    send_command(sock, TELLO_ADDRESS, 'up 50')
    time.sleep(5)

# Function to handle the main video capture and processing loop
def video_capture_loop(sock, TELLO_ADDRESS, TELLO_CAMERA_ADDRESS):
    cap = cv2.VideoCapture(TELLO_CAMERA_ADDRESS)
    if not cap.isOpened():
        cap.open(TELLO_CAMERA_ADDRESS)

    frame_num = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None or frame.size == 0:
                print("Error reading frame or empty frame received.")
                continue

            print(frame_num)

            if frame_num != 0:
                frame_num += 1
                if frame_num > 4:
                    frame_num = 0
                continue

            frame_height, frame_width = frame.shape[:2]
            frame_center_x, frame_center_y = int(frame_width / 2), int(frame_height / 2)

            cv2.circle(frame, (frame_center_x, frame_center_y), 10, (255, 0, 0), -1)

            cv2.imshow('Tello Camera View', frame)

            results = model.track(frame)  # Perform tracking
            if results:
                for result in results:
                    classes_names = result.names if hasattr(result, 'names') else []
                    if result.boxes:
                        for box in result.boxes:
                            if hasattr(box, 'conf') and box.conf is not None and box.conf[0] > 0.:
                                if hasattr(box, 'xyxy') and box.xyxy is not None:
                                    [x1, y1, x2, y2] = box.xyxy[0]
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                    cls = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else -1
                                    class_name = classes_names[cls] if cls != -1 and cls < len(classes_names) else 'Unknown'

                                    # detect_center_xとdetect_center_yの計算
                                    detect_center_x = int(((x2 - x1) / 2) + x1)
                                    detect_center_y = int(((y2 - y1) / 2) + y1)

                                    cv2.circle(frame, (detect_center_x, detect_center_y), 10, (255, 255, 255), -1)

                                    # x_center_gapとy_center_gapの計算
                                    x_center_gap = frame_center_x - detect_center_x
                                    y_center_gap = frame_center_y - detect_center_y

                                    cv2.arrowedLine(frame, (detect_center_x, detect_center_y), (frame_center_x, frame_center_y), (0, 255, 0), 3)

                                    # Get the track ID
                                    track_id = box.id[0] if hasattr(box, 'id') and box.id is not None else None

                                    # Draw the rectangle and track ID
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                                    label = f'{class_name} {box.conf[0]:.2f}'
                                    if track_id is not None:
                                        label += f' ID:{track_id}'
                                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                                    # draw box height and width
                                    box_height, box_width = int(x2 - x1), int(y2 - y1)
                                    cv2.putText(frame, f'Height: {box_height}, Width: {box_width}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                                    if (track_id == 1):
                                        if (box_width < 150):
                                            if (box_height < 80):
                                                send_command(sock, TELLO_ADDRESS, 'forward 20')
                                            else:
                                                send_command(sock, TELLO_ADDRESS, 'forward 80')
                                        elif (box_width > 200):
                                            if (box_width > 350):
                                                send_command(sock, TELLO_ADDRESS, 'backward 30')
                                            else:
                                                send_command(sock, TELLO_ADDRESS, 'backward 50')

                                        if (x_center_gap > 40):

                                            if (box_width > 250):
                                                send_command(sock, TELLO_ADDRESS, 'ccw 5')
                                            elif (box_width > 100):
                                                send_command(sock, TELLO_ADDRESS, 'ccw 10')
                                            else:
                                                send_command(sock, TELLO_ADDRESS, 'ccw 20')
                                        elif (x_center_gap < -40):
                                            if (box_width > 250):
                                                send_command(sock, TELLO_ADDRESS, 'cw 5')
                                            elif (box_width > 100):
                                                send_command(sock, TELLO_ADDRESS, 'cw 10')
                                            else:
                                                send_command(sock, TELLO_ADDRESS, 'cw 20')
                                        
                                        if (y_center_gap > 150):
                                            send_command(sock, TELLO_ADDRESS, 'up 20')
                                        elif (y_center_gap < -150):
                                            send_command(sock, TELLO_ADDRESS, 'down 20')

            cv2.imshow('Tello Camera View', frame)

            frame_num += 1

            # Add a small delay to ensure the loop runs smoothly
            if cv2.waitKey(1) & 0xFF == ord('q'):
                send_command(sock, TELLO_ADDRESS, 'land')
                break
            else:
                cv2.waitKey(1)  # Add this line to ensure proper event processing
    except KeyboardInterrupt:
        sock.sendto('land'.encode('utf-8'), TELLO_ADDRESS)
        print("Interrupted by user")
    except Exception as e:
        sock.sendto('land'.encode('utf-8'), TELLO_ADDRESS)
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.sendto('streamoff'.encode('utf-8'), TELLO_ADDRESS)
        sock.close()
        print("Socket closed")

# Main function
def main():
    TELLO_IP = '192.168.10.1'
    TELLO_PORT = 8889
    TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)
    TELLO_CAMERA_ADDRESS = 'udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=50000000'

    sock = initialize_socket(TELLO_PORT)

    udp_thread = threading.Thread(target=udp_receiver, args=(sock, ))
    udp_thread.daemon = True
    udp_thread.start()

    takeoff_thread = threading.Thread(target=takeoff, args=(sock, TELLO_ADDRESS, ))
    takeoff_thread.daemon = True
    takeoff_thread.start()

    # Run the video capture loop in the main thread
    video_capture_loop(sock, TELLO_ADDRESS, TELLO_CAMERA_ADDRESS)

if __name__ == "__main__":
    main()
