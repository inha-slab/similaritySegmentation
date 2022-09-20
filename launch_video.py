# detectoron 환경 가능하게 파이썬 환경 변경하였습니다. - DonghwiCho 2022/08/02 13:36
import argparse
import multiprocessing as mp
import pathlib
import random
import time
import json

import cv2
import detectron2.data.transforms as T
import torch
from alfred_c.vis.image.det import visualize_det_cv2_part
from alfred_c.vis.image.mask import vis_bitmasks_with_classes
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.structures.masks import BitMasks
from detectron2.utils.logger import setup_logger

from yolov7.config import add_yolo_config

# constants
WINDOW_NAME = "COCO detections"


class DefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            print("image after transform: ", image.shape)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            tic = time.time()
            # predictions, pure_t = self.model([inputs])
            predictions = self.model([inputs])
            predictions = predictions[0]
            c = time.time() - tic
            print("cost: {}, fps: {}".format(c, 1 / c))
            return predictions


def setup_cfg(args):
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.YOLO.CONF_THRESHOLD = args.confidence_threshold
    cfg.MODEL.YOLO.NMS_THRESHOLD = args.nms_threshold
    cfg.MODEL.YOLO.IGNORE_THRESHOLD = 0.1
    # force devices based on user device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MAX_SIZE_TEST = 600  # 90ms
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/inha/anaconda3/envs/scene39/lib/python3.9/site-packages/detectron2/model_zoo/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        # Donghwi Modify (Directory Error : Absoulute Directory to fix) Using Own's detectron(!Important!)
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument(
        "-i",
        "--input",
        # nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "-v",
        "--video-input",
        # nargs="+",
        help="Using video as a input; "
    )
    parser.add_argument(
        "-o",
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "-c",
        "--confidence-threshold",
        type=float,
        default=0.21,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "-n",
        "--nms-threshold",
        type=float,
        default=0.6,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Name of Weights & Biases Project.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Name of Weights & Biases Entity.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def vis_res_fast(res, img, class_names, colors, thresh):
    ins = res["instances"]
    bboxes = None
    if ins.has("pred_boxes"):
        bboxes = ins.pred_boxes.tensor.cpu().numpy()
    scores = ins.scores.cpu().numpy()
    clss = ins.pred_classes.cpu().numpy()
    if ins.has("pred_bit_masks"):
        bit_masks = ins.pred_bit_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
        # img = vis_bitmasks_with_classes(img, clss, bit_masks)
        # img = vis_bitmasks_with_classes(img, clss, bit_masks, force_colors=colors, mask_border_color=(255, 255, 255), thickness=2)
        img = vis_bitmasks_with_classes(
            img, clss, bit_masks, force_colors=None, draw_contours=True, alpha=0.8
        )

    if ins.has("pred_masks"):
        bit_masks = ins.pred_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
        img = vis_bitmasks_with_classes(
            img,
            clss,
            bit_masks,
            force_colors=None,
            draw_contours=True,
            alpha=0.6,
            thickness=2,
        )
    thickness = 1 if ins.has("pred_bit_masks") else 2
    font_scale = 0.3 if ins.has("pred_bit_masks") else 0.4
    if bboxes is not None:
        img = visualize_det_cv2_part(
            img,
            scores,
            clss,
            bboxes,
            class_names=class_names,
            force_color=colors,
            line_thickness=thickness,
            font_scale=font_scale,
            thresh=thresh,
        )
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img


def rule3(current_frame, previous_frame):
    current_hist = cv2.calcHist([current_frame], [0], None, [256], [0, 256])
    previous_hist = cv2.calcHist([previous_frame], [0], None, [256], [0, 256])

    sum = 0
    for i in range(0, 256):
        if max(current_hist[i], previous_hist[i]) == 0:
            sum = sum + 0
        else:
            hist_similarity = 1 - (abs(current_hist[i] - previous_hist[i]) / max(current_hist[i], previous_hist[i]))
            sum = sum + hist_similarity

    return sum / 256

similar3 = {}
similar3['similarity3'] = []

if __name__ == "__main__":
    #
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    class_names = cfg.DATASETS.CLASS_NAMES
    predictor = DefaultPredictor(cfg)

    print(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

    colors = [
        [random.randint(0, 255) for _ in range(3)]  # class별 랜덤 색상리스트
        for _ in range(cfg.MODEL.YOLO.CLASSES)
    ]
    conf_thresh = cfg.MODEL.YOLO.CONF_THRESHOLD
    print("confidence thresh: ", conf_thresh)

    # iter = ImageSourceIter(args.input)
    # if args.wandb_project is not None:
    #     from wandadb.wandb_logger import WandbInferenceLogger
    #
    #     inference_logger = WandbInferenceLogger(
    #         wandb_entity=args.wandb_entity,
    #         wandb_project=args.wandb_project,
    #         conf_threshold=args.confidence_threshold,
    #         config=cfg,
    #     )
    # else:
    #     inference_logger = None

    # frame, Shot count initialize
    shot = 1
    frame = 0

    # JSON Read
    frame_list = None
    with open("MiSang_Frame.json", 'r') as outfile:
        frame_list = json.load(outfile)

    # Save Shot (odd = Last_Frame, even = First_Frame)
    save_shot = []

    # Video Capture Point
    cap = cv2.VideoCapture(args.video_input)

    while cap.isOpened():
        print(f'Current Frame : {frame}')
        ret, im = cap.read()
        if not ret:
            break

        # Rule3 Point
        if frame == frame_list[shot - 1]["frame"] - 1:  # First_Frame Save Point
            save_shot.append(im)

            # First_Frame일 경우 이전 5개 샷의 Last_Frame을 체크해서 유사도를 구한다.
            if shot != 1:  # 첫번째 샷은 무시한다.
                start_point = shot - 5
                if start_point <= 1:
                    start_point = 1

                for i in range(start_point, shot):
                    print(f'Rule3 - Shot{i}, Shot{shot} : {rule3(im, save_shot[(i - 1) * 2 + 1]) * 100}%')
                    if i == (shot-1): # shot n-1, shot n save
                        similar3['similarity3'].append(float(rule3(im, save_shot[(i - 1) * 2 + 1])))
                        with open('similarity3.json', 'w', encoding="utf-8") as outfile:
                            json.dump(similar3, outfile)


            shot = shot + 1

        elif frame == frame_list[shot - 1]["frame"] - 2:  # Last_Frame Save Point
            save_shot.append(im)

        frame = frame + 1

        res = predictor(im)

        res = vis_res_fast(res, im, class_names, colors, conf_thresh)

        cv2.imshow("frame", res)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
