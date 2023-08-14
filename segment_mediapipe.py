import cv2
import mediapipe as mp
import numpy as np
import PySimpleGUI as sg
import io
from PIL import Image

# MediaPipeのSelfie Segmentationモデルを読み込み
mp_selfie_segmentation = mp.solutions.selfie_segmentation
# selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# 動画の読み込み
# cap = cv2.VideoCapture('assets/input_video_1.mp4')
# cap = cv2.VideoCapture('assets/input_video_2.mp4')
cap = cv2.VideoCapture('assets/input_video_3.mp4')

# PySimpleGUIのウィンドウを作成
sg.theme('Reddit')
layout = [[sg.Image(filename='', key='-IMAGE-')]]
window = sg.Window('Video', layout, size=(800, 600))  # ウィンドウのサイズを変更

frame_cnt = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame_cnt += 1
    if not frame_cnt % 5 == 0:  # 一定の割合で処理をスキップ
        continue
    # 元のフレームをRGBに変換
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
        break

    # フレームの前処理と推論
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = selfie_segmentation.process(frame_rgb)

    # 推論結果をマスクとして用い、元のフレームから人物部分だけを切り出す
    mask = result.segmentation_mask > 0.1

    # アルファチャンネルの作成
    alpha = np.ones(mask.shape, dtype=np.uint8) * 255  # 全てのピクセルを不透明にする
    alpha[~mask] = 0  # マスクが0の部分（人物以外の部分）を透明にする

    # 元のフレームとアルファチャンネルを結合
    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_bgra[..., 3] = alpha

    # 結果をPNGに変換
    result = cv2.resize(frame_bgra, (800, 600))  # 結果のサイズをウィンドウのサイズに合わせて変更
    img = Image.fromarray(result)
    bio = io.BytesIO()
    img.save(bio, format='PNG')
    imgbytes = bio.getvalue()

    # 結果の表示
    event, values = window.read(timeout=25)
    if event == sg.WINDOW_CLOSED:
        break
    window['-IMAGE-'].update(data=imgbytes)

# 動画ファイルの読み込みを終了し、開いたウィンドウを全て閉じる
cap.release()
window.close()
