# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
# yKesamaru added and updated the code.

from functools import partial
from pathlib import Path
import cv2
import PySimpleGUI as sg
import numpy as np
import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from examples.utils import write_mot_results


def on_predict_start(predictor, persist=False):
    # トラッカーの初期化
    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):
    # YOLOトラッキングの実行
    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz
    )


    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args

    # PySimpleGUIのウィンドウ設定
    if args.show_tk:
        layout = [[sg.Image(filename='', key='-IMAGE-')]]
        window = sg.Window('YOLO Tracking', layout, location=(800, 400))


    for frame_idx, r in enumerate(results):
        if r.boxes.data.shape[1] == 7:

            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p

        processed_img = r.orig_img.copy()
        
        for box in r.boxes.data:
            x1, y1, x2, y2, id, confidence, class_id = map(float, box)
            x1, y1, x2, y2, id, class_id = map(int, [x1, y1, x2, y2, id, class_id])  # 整数型に変換
            confidence = round(confidence * 100)  # 小数点以下2桁までの表示にするため100倍して四捨五入
            cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # クラス名のマッピング（この例では0がpersonに対応）
            class_names = {0: "hito"}
            class_name = class_names.get(class_id, "")

            # テキストの描画（クラス名、信頼度など）
            text = f"id:{id} {class_name} {confidence}%"
            cv2.putText(processed_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if args.show_tk:
            event, values = window.read(timeout=0)
            imgbytes = cv2.imencode('.png', processed_img)[1].tobytes()
            window['-IMAGE-'].update(data=imgbytes)
            
            if event == sg.WINDOW_CLOSED:
                break


def parse_opt():
    # 引数の設定
    class Args:
        def __init__(self):
            self.yolo_model = Path(WEIGHTS / 'yolov8n')
            self.reid_model = Path(WEIGHTS / 'osnet_x0_25_msmt17.pt')
            self.tracking_method = 'deepocsort'
            self.source = 'assets/input_video_4.mp4'
            self.imgsz = [640]
            self.conf = 0.5
            self.iou = 0.7
            self.device = '0'
            
            self.show_tk =  True

            self.show = False
            self.save = True
            self.classes = [0]
            self.project = ROOT / 'runs' / 'track'
            self.name = 'exp'
            self.exist_ok = False
            self.half = False
            self.vid_stride = 4
            self.show_labels = True
            self.show_conf = True
            self.save_txt = False
            self.save_id_crops = False
            self.save_mot = False
            self.line_width = 1
            self.per_class = False
            self.verbose = True

    return Args()


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
