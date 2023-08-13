import cv2
import numpy as np
import PySimpleGUI as sg
from sklearn.cluster import DBSCAN

# 人物検出のための準備
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ビデオの読み込み
video_path = 'assets/input_video_4.mp4'
cap = cv2.VideoCapture(video_path)

# PySimpleGUIのウィンドウ設定
layout = [[sg.Image(filename='', key='-IMAGE-')],
          [sg.Button('Exit', size=(10, 1))]]

window = sg.Window('DBSCAN People Detection', layout, location=(800, 400))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 人物検出
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

    # DBSCANの適用
    if len(boxes) > 0:
        dbscan = DBSCAN(eps=50, min_samples=1)
        clusters = dbscan.fit_predict(boxes)

        # クラスタごとに短形で囲む
        for cluster_id in np.unique(clusters):
            cluster_boxes = boxes[clusters == cluster_id]
            x_min = cluster_boxes[:, 0].min()
            y_min = cluster_boxes[:, 1].min()
            x_max = cluster_boxes[:, 0].max() + cluster_boxes[:, 2].max()
            y_max = cluster_boxes[:, 1].max() + cluster_boxes[:, 3].max()
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # PySimpleGUIでの表示
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)

    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break

cap.release()
window.close()
