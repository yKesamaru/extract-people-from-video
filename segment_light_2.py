import io

import cv2
import numpy as np
import PySimpleGUI as sg
from PIL import Image, ImageTk

# cap = cv2.VideoCapture('assets/input_video_1.mp4')
# cap = cv2.VideoCapture('assets/input_video_2.mp4')
cap = cv2.VideoCapture('assets/input_video_3.mp4')

# 最初のフレームを読み込む
ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (800, 600))
# グレースケールに変換
background = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# PySimpleGUIの設定
sg.theme('Reddit')
layout = [[sg.Image(filename='', key='-IMAGE-')]]
window = sg.Window('Image', layout, location=(800,400))

frame_cnt = 0
while True:
    # 2つ目のフレームを読み込む
    ret, frame2 = cap.read()
    frame_cnt += 1
    if not frame_cnt % 5 == 0:  # 一定の割合で処理をスキップ
        continue
    frame2 = cv2.resize(frame2, (800, 600))
    if ret == False:
        break
    # グレースケールに変換
    current = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # フレーム間の違い（動き）を計算する
    diff = cv2.absdiff(background, current)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # 4チャンネルの画像に変換（B, G, R, A）
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2BGRA)

    # 動いていない部分を透明にする
    frame2[..., 3] = mask

    # PNG形式にエンコード
    is_success, buffer = cv2.imencode(".png", frame2)
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
