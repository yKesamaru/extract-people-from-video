# はじめに

人物抽出は、画像や動画から特定の人物を背景から分離する技術です。
AR（拡張現実）やVR（仮想現実）のようなアプリケーションでよく使用されます。

今回は導入として、OpenCV、MediaPipe、DeepLabV3のResNet101とMobileNetV3 Largeの異なる手法を使用して、人物抽出のコードを考察します。


人物が歩いている動画から、人物だけを切り取り背景を無くす手法は主に「背景差分法」と「セマンティックセグメンテーション」の2つがあります。

- 背景差分法：動画の最初のフレームを背景として設定し、その後のフレームと比較して変化があった部分を抽出します。これは人物が動いている場合に有効ですが、背景自体が動いている場合や複数の人物が動いている場合には適用が難しいです。
- セマンティックセグメンテーション：画像内の各ピクセルが何のオブジェクトに属しているかを予測する深層学習の手法です。

# 環境
```bash
Python 3.8.10
(FACE01) 
$ inxi -SCGxx --filter
System:    Kernel: 5.15.0-46-generic x86_64 bits: 64 compiler: N/A Desktop: Unity wm: gnome-shell dm: GDM3 
           Distro: Ubuntu 20.04.4 LTS (Focal Fossa) 
CPU:       Topology: Quad Core model: AMD Ryzen 5 1400 bits: 64 type: MT MCP arch: Zen rev: 1 L2 cache: 2048 KiB 
Graphics:  Device-1: NVIDIA TU116 [GeForce GTX 1660 Ti] vendor: Micro-Star MSI driver: nvidia v: 515.65.01 bus ID: 08:00.0 
```


# 方法
元動画

https://pixabay.com/ja/

![](https://raw.githubusercontent.com/yKesamaru/extract-people-from-video/master/assets/original.gif)

## OpenCV
OpenCVは、コンピュータビジョンのためのオープンソースライブラリで、画像処理や機械学習などの機能を提供しています。
様々な機能がてんこ盛りなライブラリですが、今回は3種類を試したいと思います。

### 「光学的フロー(Optical Flow)」を用いた動き検出
光学的フローは、ビデオや画像シーケンスにおける物体やカメラの動きを推定するための手法です。このコードでは、その光学的フローを用いて動きのある部分を検出し、動きのない部分を透明化しています。
```python
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
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

# PySimpleGUIの設定
sg.theme('Reddit')
layout = [[sg.Image(filename='', key='-IMAGE-')]]
window = sg.Window('Image', layout, location=(800,400))

frame_cnt = 0
while True:
    # 2つ目のフレームを読み込む
    ret, frame2 = cap.read()
    frame_cnt += 1
    if not frame_cnt % 2 == 0:  # 一定の割合で処理をスキップ
        continue
    frame2 = cv2.resize(frame2, (800, 600))
    if ret == False:
        break
    # グレースケールに変換
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # フレーム間の違い（動き）を計算する
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 動きを極座標に変換し、色と強度で表示する
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # 動きの強度に基づいて二値化する
    _, mask = cv2.threshold(mag, 1.0, 255, cv2.THRESH_BINARY)

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

    # 次のフレームを読み込むために現在のフレームを更新
    prvs = next

    if event == sg.WINDOW_CLOSED:
        break
window.close()
cap.release()


```

![](https://raw.githubusercontent.com/yKesamaru/extract-people-from-video/master/assets/2.gif)

### 背景除去（または背景分離）
通常の背景除去の手法は、背景が静止していて変化しないという前提に基づいています。つまり、背景はカメラが動かない限り一定で、動いている物体だけが前景となります。このため、最初の数フレームを使って背景モデルを作成し、その後のフレームで背景モデルと大きく異なる部分を前景（動いている物体）として検出します。

具体的なアルゴリズムとしては、単純なものではフレーム間の差分を取る方法、少し複雑なものではガウシアン混合モデル（Gaussian Mixture Model, GMM）を用いる方法などがあります。

```python
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

```
![](https://raw.githubusercontent.com/yKesamaru/extract-people-from-video/master/assets/1.gif)

### ガウシアン混合モデル（Gaussian Mixture Model, GMM）を用いて動きを検出する
cv2.createBackgroundSubtractorMOG2()を用いてガウシアン混合モデルによる背景差分法を初期化し、各フレームに対してfgbg.apply(frame)を適用して前景マスクを取得しています。このマスクを用いて動いていない部分を透明にし、動いている部分だけを表示しています。
```python
import io

import cv2
import numpy as np
import PySimpleGUI as sg
from PIL import Image, ImageTk

cap = cv2.VideoCapture('assets/input_video_3.mp4')

# ガウシアン混合モデルによる背景差分法を初期化
fgbg = cv2.createBackgroundSubtractorMOG2()

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
    frame = cv2.resize(frame, (800, 600))
    if ret == False:
        break

    # ガウシアン混合モデルによる背景差分法を適用
    fgmask = fgbg.apply(frame)

    # 4チャンネルの画像に変換（B, G, R, A）
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # 動いていない部分を透明にする
    frame[..., 3] = fgmask

    # PNG形式にエンコード
    is_success, buffer = cv2.imencode(".png", frame)
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

```

![](https://raw.githubusercontent.com/yKesamaru/extract-people-from-video/master/assets/ガウシアン.gif)


## MediaPipe

MediaPipeは、Googleが開発したマルチモーダル（音声、ビデオ、センサーデータなど）の機械学習パイプラインの構築を支援するフレームワークです。

```python
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

```


![](https://raw.githubusercontent.com/yKesamaru/extract-people-from-video/master/assets/mediapipe1.gif)


## DeepLabV3 (ResNet101とMobileNetV3 Large)

DeepLabV3は、セマンティックセグメンテーション（画像の各ピクセルに対してラベルを割り当てるタスク）のための深層学習モデルです。ResNet101とMobileNetV3 Largeは、DeepLabV3のバックボーン（主要な特徴抽出部分）として使用されるネットワークです。
```python
import io

import cv2
import numpy as np
import PySimpleGUI as sg
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

# モデルの読み込み
# model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
# MobileNetV2をバックボーンに持つDeepLabV3モデルの読み込み
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
```

![](https://raw.githubusercontent.com/yKesamaru/extract-people-from-video/master/assets/deeplab_mobilenet.gif)



# 考察
今回試した全ての結果が、満足の行くものではありませんでした。
以下に、それぞれの手法の特徴をまとめます。

## OpenCV

OpenCVでは、光学的フロー、単純な背景差分法、ガウシアン混合モデル（Gaussian Mixture Model, GMM）のそれぞれを実験しました。
特定の状況下では有用ですが、複雑な背景や照明条件下では、人物抽出の精度が低下しています。

## MediaPipe

MediaPipeの人物抽出は、そもそもWEBカメラの真ん前に座る人物に対して有効です。そのため、一般的な距離の場合の精度はかなり落ちます。処理そのものは軽い印象を受けました。

## DeepLabV3 MobileNetV3 Large

DeepLabV3 ResNet101は、人物抽出の精度が高いですが、その処理速度はとても遅く、実用に耐えません。今回はMobileNetをベースとした学習モデルを使用しました。

DeepLabV3 MobileNetV3 Largeは、精度と速度のバランスが良いです。リアルタイムのアプリケーションでもなんとか可能、複雑な背景や照明条件下でも比較的良好な精度でした。

# 結論
以上です。
今回の人物抽出では良好な結果は得られませんでしたが、手法は他にも存在します。
今回は「その①」ということで、追加実験を行う予定です。

「人物抽出のためのコード比較その②: YOLOv5, SORT」に続きます。
https://zenn.dev/ykesamaru/articles/6cb451f8fd1740

<その②のちょい見せ>
![](https://raw.githubusercontent.com/yKesamaru/extract-people-from-video/master/assets/yolov5_sort.gif)