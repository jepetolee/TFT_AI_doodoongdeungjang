import argparse
import tensorflow as tf
import tensorflow.keras.layers as layer
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import math
import cv2
import numpy as np
from PIL import ImageGrab,Image, ExifTags
from pathlib import Path
import glob
import random
import shutil
import time
from threading import Thread
import pyautogui
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision
from YOLOv5writtencode.data import *

def findthescreen():  # 게임 등수, 회전초밥, 게임 재시작 화면 1, 게임 재시작 화면 2, 게임 선택창 ,일반
    screen = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 1080)))
    model = tf.keras.models.load_model('C:/Users/user/PycharmProjects/BRAIN.net/YOLOv5writtencode/model/save.h5')
    image_array = np.asarray(screen)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction


def checktheboxes():  # 게임 내 박스 파일을 조사하여 캐릭터 배치 현황을 이해하는 함수
    out, weights, imgsz = \
        opt.output, opt.weights, 1920

    # Initialize
    device = select_device('cpu')
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader

    save_img = True
    pic= pyautogui.screenshot(region=(0,0,1920,1080))
    image_data= np.array(pic)
    image_data=cv2.cvtColor(image_data,cv2.COLOR_RGB2BGR)

    imagedata=letterbox(image_data,new_shape=1920)[0]

    imagedata = imagedata[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    imagedata = np.ascontiguousarray(imagedata)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, 1920, 1080), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    img = torch.from_numpy(imagedata).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 =time_synchronized()
    pred = model(img, augment='store_true')[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
    t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):
        s =  ''
        s += '%gx%g ' % img.shape[2:]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
             # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

        print('%sDone. (%.3fs)' % (s, t2 - t1))


def check_numbers():
    global enviroment
    device = torch.device('cpu')  # 게임내에서는 cud 0으로 바꾸자!
    saved_model_path = 'yolov5xnum.pt'
    model = attempt_load(weights=saved_model_path, map_location=device)

    screen = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 1080)))
    dataset = LoadImages(screen)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    img = torch.zeros((1, 3, 1920, 1080), device=device)
    _ = model(img) if device.type != 'cpu' else None
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment='store_true')[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        enviroment = int(cls)

    return enviroment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    opt = parser.parse_args()
    checktheboxes()
