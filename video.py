from ultralytics import YOLO
from PIL import Image
import cv2
import csv
import os

output_dir = "runs/detect/predict"
os.makedirs(output_dir, exist_ok=True)

csv_filename = os.path.join(output_dir, 'sample1.csv')
# 出力するデータ
header = [
    'No', 'Class', 'Label', 'Scores', 'id', 'x1', 'y1', 'x2', 'y2',
    'detect_center_x', 'detect_center_y', 'frame_center_x', 'frame_center_y',
    'x_center_gap', 'y_center_gap'
]
# CSVファイルにデータを書き込む
file = open(csv_filename, mode='w', newline='', encoding='utf-8')
writer = csv.writer(file)
writer.writerow(header)

# 動画ファイルを開く
model = YOLO("./YOLOv8-HumanDetection/best.pt")

video_path = "usevideo/test.mp4"
output_video_path = os.path.join(output_dir, "output_test.mp4")

cap = cv2.VideoCapture(video_path)

# ビデオライターの設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画のコーデック
fps = cap.get(cv2.CAP_PROP_FPS)  # 元の動画のフレームレート
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 元の動画の幅
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 元の動画の高さ
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_center_x = width / 2
frame_center_y = height / 2

frame_cnt = 0
# 動画のフレームをループ処理
while cap.isOpened():
    # 動画からフレームを読み取る
    success, frame = cap.read()

    if success:
        # フレームに対してYOLOv8の推論を実行
        results = model.track(frame, persist=True, conf=0.5, classes=[0, 2, 7])

        frame_cnt += 1

        # 結果をフレームに可視化
        annotated_frame = results[0].plot()

        items = results[0]
        for item in items:
            cls = int(item.boxes.cls)  # クラスID, (N, 1)
            label = item.names[int(cls)]
            score = item.boxes.conf.cpu().numpy()[0]  # 信頼度スコア, (N, 1)
            x1, y1, x2, y2 = item.boxes.xyxy.cpu().numpy()[0]  # ボックスのxyxy形式, (N, 4)

            id_value = item.boxes.id
            if id_value is None:
                track_ids = ''
            else:
                track_ids = item.boxes.id.int().cpu().tolist()[0]

            # detect_center_xとdetect_center_yの計算
            detect_center_x = (x1 + x2) / 2
            detect_center_y = (y1 + y2) / 2

            # x_center_gapとy_center_gapの計算
            x_center_gap = frame_center_x - detect_center_x
            y_center_gap = frame_center_y - detect_center_y

            csv_data = [
                str(frame_cnt), str(cls), str(label), str(score), str(track_ids),
                str(x1), str(y1), str(x2), str(y2), str(detect_center_x), str(detect_center_y),
                str(frame_center_x), str(frame_center_y),
                str(x_center_gap), str(y_center_gap)
            ]
            writer.writerow(csv_data)

        # 注釈付きフレームを保存
        out.write(annotated_frame)

        # 注釈付きフレームを表示
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 動画の終わりに達したらループを抜ける
        break

# ビデオキャプチャオブジェクトとビデオライターオブジェクトを解放し、表示ウィンドウを閉じる
cap.release()
out.release()
cv2.destroyAllWindows()

# ファイルを閉じる
file.close()
