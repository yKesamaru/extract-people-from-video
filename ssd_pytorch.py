import cProfile

def main():
    import cv2
    import PySimpleGUI as sg
    import torch
    from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_predictor
    from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの読み込み
    model_path = 'models/mb2-ssd-lite-mp-0_686.pth'
    class_labels_path = 'models/voc-model-labels.txt'
    with open(class_labels_path, 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]

    # MobileNetV2 SSD Lite のモデルを作成
    net = create_mobilenetv2_ssd_lite(len(class_labels), is_test=True)
    net.load(model_path)
    net = net.to(device)  # モデルをデバイスに移動
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

    # PySimpleGUIのウィンドウを作成
    layout = [[sg.Image(filename='', key='image')],
            [sg.Button('Exit')]]

    window = sg.Window('Object Detection', layout, location=(800, 400))

    # ビデオの読み込み
    video_path = 'assets/input_video_4.mp4'
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 推論の実行
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device)
        frame_numpy = frame_tensor[0].permute(1, 2, 0).cpu().numpy()  # テンソルをnumpy配列に変換
        boxes, labels, probs = predictor.predict(frame_numpy, 10, 0.4)  # numpy配列を渡す

        # 検出結果の処理
        for i in range(boxes.size(0)):
            box = boxes[i, :].cpu().numpy()  # テンソルをCPUに移動し、NumPy配列に変換
            label = f"{class_labels[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)


        # 結果の表示（PySimpleGUI）
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)

        event, values = window.read(timeout=20)
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

    cap.release()
    cv2.destroyAllWindows()
    window.close()


if __name__ == '__main__':
    cProfile.run('main()', 'profile_result.prof')