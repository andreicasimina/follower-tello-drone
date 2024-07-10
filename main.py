from ultralytics import YOLO
from PIL import Image
import cv2
import socket
import threading
import time
import numpy as np
#import os


#YOLO初期化
# YOLO modelの初期化i
#model = YOLO("yolov8n.pt")

# bus.jpgのパス
#image_path = "usevideo/bus.jpg"

# 画像をPIL Imageとして読み込む
#im1 = Image.open(image_path)

# YOLOモデルで予測を実行し、保存する
#results = model.predict(source=im1, save=True)

# 保存された画像のパスを表示する
#print("保存された画像:", results)

##mp4ファイルの動画の物体認識  
model = YOLO("./YOLOv8-HumanDetection/best.pt")

#cuda使うのはたぶんこれ
#model.to("cuda")

image_path = "usevideo/test.mp4"



#ドローン制御部
# データ受け取り用の関数
def udp_receiver():
    while True:
        try:
            response, _ = sock.recvfrom(1024)
        except Exception as e:
            print(e)
            break

# Tello側のローカルIPアドレス(デフォルト)、宛先ポート番号(コマンドモード用)
TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)

# Telloからの映像受信用のローカルIPアドレス、宛先ポート番号
TELLO_CAMERA_ADDRESS = 'udp://@0.0.0.0:11111'
# TELLO_CAMERA_ADDRESS = '192.168.10.1:11111'

# キャプチャ用のオブジェクト
cap = None

# データ受信用のオブジェクト備
response = None

# 通信用のソケットを作成
# ※アドレスファミリ：AF_INET（IPv4）、ソケットタイプ：SOCK_DGRAM（UDP）
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 自ホストで使用するIPアドレスとポート番号を設定
sock.bind(('', TELLO_PORT))

# 受信用スレッドの作成
thread = threading.Thread(target=udp_receiver, args=())
thread.daemon = True
thread.start()

# コマンドモード
sock.sendto('command'.encode('utf-8'), TELLO_ADDRESS)

time.sleep(1)

# カメラ映像のストリーミング開始
sock.sendto('streamon'.encode('utf-8'), TELLO_ADDRESS)

time.sleep(5)

if cap is None:
    cap = cv2.VideoCapture(TELLO_CAMERA_ADDRESS)

if not cap.isOpened():
    cap.open(TELLO_CAMERA_ADDRESS)

time.sleep(1)

s = 0

while True:
    ret, frame = cap.read()

    # 動画フレームが空ならスキップ
    if frame is None or frame.size == 0:
        continue

    # カメラ映像のサイズを半分にしてウィンドウに表示
    frame_height, frame_width = frame.shape[:2]
    frame = cv2.resize(frame, (int(frame_width/2), int(frame_height/2)))
    
    cv2.imshow('Tello Camera View', frame)
    cv2.imwrite('lena_opencv_red.jpg', frame)
    s = s + 1

    image_path = "/Users/masataka/Desktop/PBL/AutoDrone/lena_opencv_red.jpg"

    # WEBカメラからリアルタイム検出
    results = model(source=image_path, show=True)
    for i in enumerate(results):
        print(i)

    # 参照ファイル: https://ai-wonderland.com/entry/yolov8webcamera

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        break
cap.release()
cv2.destroyAllWindows()

# ビデオストリーミング停止
sock.sendto('streamoff'.encode('utf-8'), TELLO_ADDRESS)