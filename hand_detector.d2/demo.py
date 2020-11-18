# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""


import logging
import os
from collections import OrderedDict
from tqdm import tqdm
import torch
import json

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
import cv2
import random

from detectron2.data.datasets.pascal_voc import register_pascal_voc
import numpy as np

_datasets_root = "datasets"
for d in ["trainval", "test"]:
    register_pascal_voc(name=f'100DOH_hand_{d}', dirname=_datasets_root, split=d, year=2007, class_names=["hand"])
    MetadataCatalog.get(f'100DOH_hand_{d}').set(evaluator_type='pascal_voc')

import sys
sys.path.append('..')
from utils import gen_json

if __name__ == '__main__':

    # load cfg and model
    cfg = get_cfg()
    cfg.merge_from_file("faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
    cfg.MODEL.WEIGHTS = 'models/model_0529999.pth' # add model weight here
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 0.5 , set the testing threshold for this model

    # data path
    # test_img = './viz/input.jpg'
    imgs = open('../data/vlog_imgs.json')
    test_imgs = json.load(imgs)
    save_dir = '../100doh_viz'
    os.makedirs(save_dir, exist_ok=True)

    # predict
    predictor = DefaultPredictor(cfg)

    # output
    bboxes = []
    for img in tqdm(test_imgs):
        test_img = os.path.join('../data', img)
        im = cv2.imread(test_img)
        outputs = predictor(im)
        #  6: RWrist
        # 11: LWrist
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # [x0, y0, x1, y1]
        # print(outputs['instances'].to('cpu').pred_boxes.tensor.numpy())
        # print(outputs['instances'].to('cpu').scores.numpy())

        # cv2.imwrite(save_dir + '/100doh_' + img, v.get_image()[:, :, ::-1])
        bbox = {}
        bbox['filename'] = img
        bbox['bboxes'] = outputs['instances'].to('cpu').pred_boxes.tensor.numpy().tolist()
        bbox['score'] = outputs['instances'].to('cpu').scores.numpy().tolist()
        bboxes.append(bbox)
    gen_json.genDetecBboxes(bboxes)

    # print
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    