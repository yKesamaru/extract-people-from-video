import cv2
import PySimpleGUI as sg
import tensorflow as tf

# 利用可能なGPUデバイスを取得
gpus = tf.config.experimental.list_physical_devices('GPU')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# GPUが存在する場合、特定のGPU（例：最初のGPU）を使用するように設定
if gpus:
    try:
        # ここでは最初のGPUを使用するように設定しています
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("GPU[0]が設定されました")
    except RuntimeError as e:
        print(e)  # GPUデバイスが見つからない場合などのエラーを表示

# モデルのパス
model_path = 'ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model'

# モデルの読み込み
model = tf.saved_model.load(model_path)

# 推論関数の取得
infer = model.signatures["serving_default"]

# ウィンドウの設定
layout = [[sg.Image(filename='', key='image')]]
window = sg.Window('SSD People Detection', layout, location=(800, 400))

# ビデオの読み込み
video_path = 'assets/input_video_4.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 推論の実行
    # 推論の実行
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)  # dtypeを追加
    detections = infer(input_tensor)

    # 検出結果の処理
    boxes = detections['detection_boxes'].numpy()[0]
    scores = detections['detection_scores'].numpy()[0]
    for i, box in enumerate(boxes):
        if scores[i] > 0.5:  # 信頼度スコアが0.5以上の場合
            y1, x1, y2, x2 = map(int, box * [frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 結果の表示（PySimpleGUI）
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['image'].update(data=imgbytes)

    event, values = window.read(timeout=20)
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break

cap.release()
cv2.destroyAllWindows()
window.close()
