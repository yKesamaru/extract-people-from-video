import sys
sys.path.append('/usr/lib/python3/dist-packages')

import io
import cv2
import torch
import numpy as np
import random
import PySimpleGUI as sg
from PIL import Image, ImageTk
from sort import Sort

# YOLOv5モデルのロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# SORTのインスタンスを作成
mot_tracker = Sort()

# トラッカーIDと色のマッピングを保存する辞書
id_color_map = {}


def process_frame(frame):
    # フレームをPILイメージに変換
    img = Image.fromarray(frame)
    
    # YOLOv5で予測
    results = model(img)
    
    # 人間のクラスIDを取得（YOLOv5の場合、0が人間）
    person_class_id = 0
    
    # フレームを透明化
    frame_alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_alpha[..., 3] = 0  # 全部透明にする

    # 物体検出の結果を入力として追跡を行う
    dets = []
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == person_class_id:
            x1, y1, x2, y2 = map(int, (b.cpu() for b in box))
            dets.append([x1, y1, x2, y2, conf.cpu()]) 
    track_bbs_ids = mot_tracker.update(np.array(dets))

    for bbox in track_bbs_ids:
        x1, y1, x2, y2, track_id = map(int, bbox)

        # トラッカーIDに対応する色がまだ割り当てられていない場合は、新しい色を生成
        if track_id not in id_color_map:
            id_color_map[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # トラッカーIDに対応する色を取得
        color = id_color_map[track_id]

        # ボーダーラインを描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # トラッカーIDを表示（人物領域の内側に表示）
        cv2.putText(frame, str(track_id), (x1+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 人物部分だけ不透明にし、色を割り当てる
        frame_alpha[y1:y2, x1:x2, :3] = frame[y1:y2, x1:x2, :]
        frame_alpha[y1:y2, x1:x2, 3] = 255

    return frame_alpha


# 動画の読み込み
cap = cv2.VideoCapture('assets/input_video_3.mp4')

# PySimpleGUIの設定
sg.theme('Reddit')
layout = [[sg.Image(filename='', key='-IMAGE-')]]
window = sg.Window('Image', layout, location=(800,400))

frame_cnt = 0
while True:
    # フレームを読み込む
    ret, frame = cap.read()
    frame_cnt += 1
    if not frame_cnt % 5 == 0:  # 一定の割合で処理をスキップ
        continue
    if ret == False:
        break
    frame = cv2.resize(frame, (800, 600))

    # YOLOv5で人物以外の領域を透明化
    frame_alpha = process_frame(frame)

    # PNG形式にエンコード
    is_success, buffer = cv2.imencode(".png", frame_alpha)
    if is_success:
        # エンコードした画像をPIL.Imageに変換
        bio = io.BytesIO(buffer)
        image = Image.open(bio)

        # PySimpleGUIで表示
        event, values = window.read(timeout=25)
        if event == sg.WINDOW_CLOSED:
            break
        window['-IMAGE-'].update(data=ImageTk.PhotoImage(image))

    if event == sg.WINDOW_CLOSED:
        break
window.close()
cap.release()
