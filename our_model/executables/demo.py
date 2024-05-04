# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo as VisualizationDemo
from predictor2 import VisualizationDemo as VisualizationDemo2

#might need to install these and update playsound in terminal
from enum import Enum
import subprocess
from playsound import playsound
#import tkinter as tk

#master = tk.Tk()

class Status(Enum):
  GREEN = 1
  YELLOW = 2
  RED = 3

class P:#class vars are persistent
  status = Status.GREEN
  count = 0
  play = True

def attention_status(cur_bodies, cur_faces, thresh = 0.79, cthresh = 4, safety = 0.5):
  with open('attentionOutput.txt', 'w') as f:
    info_message = f"{cur_bodies} people detected in frame."
    print(info_message)
    f.write(info_message)
    #tk.Label(master, text=info_message).grid(row=2, column=1) 
    info_message = f"{cur_faces} people facing forward."
    print(info_message)
    f.write(info_message)
    #tk.Label(master, text=info_message).grid(row=3, column=1)

    attn_score = cur_faces/cur_bodies
    info_message = f"Paying Attention score: {attn_score}"
    print(info_message)
    f.write(info_message)
    #tk.Label(master, text=info_message).grid(row=4, column=1)

    if attn_score <= thresh:
      if P.count != cthresh:
        P.count = P.count +1
    else:
      if P.count > 0:
        P.count = P.count -1#perhaps -2

    if P.count < (cthresh * safety):
      P.status = Status.GREEN
      P.play = True
    elif P.count == cthresh:
      P.status = Status.RED
      if P.play:
        playsound('/projectnb/dl523/students/jbardwic/break_x.wav')
        P.play = False#only plays once per red trigger
    else:
      P.status = Status.YELLOW

    info_message = f"Class status: {P.status}\n"
    print(info_message)
    f.write(info_message)
    ##tk.Label(master, text=info_message).grid(row=5, column=1)

  return 0

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg_primary(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    primary_opts = args.opts[:len(args.opts)//2]
    cfg.merge_from_list(primary_opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

#get second model config
def setup_cfg_secondary(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    secondary_opts = args.opts[len(args.opts)//2:]
    cfg.merge_from_list(secondary_opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    #instantiate the models separately
    cfg = setup_cfg_primary(args)
    cfg_secondary = setup_cfg_secondary(args)

    demo = VisualizationDemo(cfg)
    demo_secondary = VisualizationDemo2(cfg_secondary)

    #tally for attention score
    people_instances = 0;
    face_instances = 0
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            people_instances += len(predictions["instances"]) if "instances" in predictions else 0
            predictions2, visualized_output2 = demo_secondary.run_on_image(img)
            face_instances +=  len(predictions2["instances"]) if "instances" in predictions2 else 0
            
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            #calculate status
            attention_status(people_instances, face_instances, thresh = 1, cthresh = 4, safety = 0.5)
            
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert (
                        len(args.input) == 1
                    ), "Please specify a directory with args.output"
                    out_filename = args.output
                #write to a file if specified
                visualized_output.save(out_filename)
                visualized_output2.save(out_filename)
            else:
                #put the overlays on input photo
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                combined_image = np.hstack((visualized_output.get_image()[:, :, ::-1], visualized_output2.get_image()[:, :, ::-1]))
                cv2.imshow(WINDOW_NAME, combined_image)
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv")
            if test_opencv_video_format("x264", ".mkv")
            else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        #overlay of the two models, unable to access prediction and calculate attention_score
        for vis_frame, vis_frame2 in tqdm.tqdm(zip(demo_secondary.run_on_video(video), demo.run_on_video(video)), total=num_frames):
          people_instances = len(predictions["instances"]) if "instances" in predictions else 0
          face_instances = len(predictions2["instances"]) if "instances" in predictions2 else 0
          attention_status(people_instances, face_instances, thresh = 1, cthresh = 4, safety = 0.5)
          if args.output:
            output_file.write(vis_frame)
          else:
            cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            combined_image = np.hstack((vis_frame, vis_frame2))
            cv2.imshow(basename, combined_image)
            if cv2.waitKey(1) == 27:
              break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()  # pragma: no cover
