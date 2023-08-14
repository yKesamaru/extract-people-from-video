import io
import cv2
import torch
import PySimpleGUI as sg
from PIL import Image, ImageTk

# YOLOv5モデルのロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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

    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == person_class_id:
            x1, y1, x2, y2 = map(int, box)
            frame_alpha[y1:y2, x1:x2, 3] = 255  # 人物部分だけ不透明にする
    
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
