import cProfile
import io

import cv2
import numpy as np
import PySimpleGUI as sg
import torch
import torchvision.transforms as T
from PIL import Image


def main():
    # モデルの読み込み
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
    model.eval()

    # 画像の前処理を定義
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

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
        if not ret:
            break
        # 元のフレームをRGBに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # フレームの前処理と推論
        input = transform(frame).unsqueeze(0)
        output = model(input)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # 推論結果をマスクとして用い、元のフレームから人物部分だけを切り出す
        mask = (output_predictions == 15).astype(np.uint8)  # 人物クラスのラベルは15, マスクをint型に変換
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # マスクをフレームの形状にリサイズ

        # アルファチャンネルの作成
        alpha = np.ones(mask.shape, dtype=np.uint8) * 255  # 全てのピクセルを不透明にする
        alpha[mask == 0] = 0  # マスクが0の部分（人物以外の部分）を透明にする

        # 元のフレームとアルファチャンネルを結合
        result = np.dstack([frame, alpha])


        # 結果をPNGに変換
        result = cv2.resize(result, (800, 600))  # 結果のサイズをウィンドウのサイズに合わせて変更
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)  # 色の順序をBGRからBGRAに変更
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

# プロファイリングの実行
# cProfile.run('main()')
# 合計時間（'tottime'）でソート
# cProfile.run('main()', sort='tottime')
# 各関数の実行時間とその関数が呼び出された回数を考慮した結果を表示
# cProfile.run('main()', sort='cumtime')

# pr = cProfile.Profile()
# pr.enable()
# main()
# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('tottime')  # 'tottime'を変更してソート方法を変えることができます
# ps.print_stats(10)  # 上位10位だけ表示
# print(s.getvalue())

cProfile.run('main()', 'profile.prof')