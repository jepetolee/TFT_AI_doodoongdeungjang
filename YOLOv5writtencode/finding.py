import torch.backends.cudnn as cudnn
import argparse
from keys import *
import torchvision.transforms as transforms
from YOLOv5writtencode.models.experimental import *
from YOLOv5writtencode.utils.datasets import *
from YOLOv5writtencode.utils.utils import *
import torch
import torch.nn.functional as F
from PIL import ImageGrab
from Pi import *
from functions import *
from Variables import *
from utils.datasets import LoadImages_IB



def check_Money():
    class CNN(torch.nn.Module):

        def __init__(self):
            super(CNN, self).__init__()
            self.keep_prob = 0.5
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

            self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            self.layer4 = torch.nn.Sequential(
                self.fc1,
                torch.nn.ReLU(),
                torch.nn.Dropout(p=1 - self.keep_prob))
            # L5 Final FC 625 inputs -> 10 outputs
            self.fc2 = torch.nn.Linear(625, 10, bias=True)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0), -1)  # Flatten them for FC
            out = self.layer4(out)
            out = self.fc2(out)
            return out

    device = torch_utils.select_device('')
    weights = 'number.pt'
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    money1 = ImageGrab.grab(bbox=(880, 860, 890, 878)).resize((28, 28)).convert("L")

    money2 = ImageGrab.grab(bbox=(890, 860, 900, 878)).resize((28, 28)).convert("L")
    model = CNN()
    checkpoint = torch.load(weights, device)
    model.load_state_dict(checkpoint, device)

    stage = transforms.ToTensor()(money1).unsqueeze(0)
    gauze = transforms.ToTensor()(money2).unsqueeze(0)
    stage = model(stage)
    gauze = model(gauze)
    stage = torch.max(stage, 1)
    stage = classes[int(stage[0])]
    gauze = torch.max(gauze, 1)
    gauze = classes[int(gauze[0])]
    money= int(stage)*10+int(gauze)
    return money


def check_Stage():
    class CNN(torch.nn.Module):

        def __init__(self):
            super(CNN, self).__init__()
            self.keep_prob = 0.5
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

            self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            self.layer4 = torch.nn.Sequential(
                self.fc1,
                torch.nn.ReLU(),
                torch.nn.Dropout(p=1 - self.keep_prob))
            # L5 Final FC 625 inputs -> 10 outputs
            self.fc2 = torch.nn.Linear(625, 10, bias=True)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0), -1)  # Flatten them for FC
            out = self.layer4(out)
            out = self.fc2(out)
            return out

    device = torch_utils.select_device('')
    weights = 'number.pt'
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    stage = ImageGrab.grab(bbox=(785, 35, 795, 55)).resize((28, 28)).convert("L")

    gauze = ImageGrab.grab(bbox=(805, 35, 815, 55)).resize((28, 28)).convert("L")
    model = CNN()
    checkpoint = torch.load(weights, device)
    model.load_state_dict(checkpoint, device)

    stage = transforms.ToTensor()(stage).unsqueeze(0)
    gauze = transforms.ToTensor()(gauze).unsqueeze(0)
    stage = model(stage)
    gauze = model(gauze)
    stage = torch.max(stage, 1)
    stage = classes[int(stage[0])]
    gauze = torch.max(gauze, 1)
    gauze = classes[int(gauze[0])]
    print(stage, gauze)
    return stage, gauze


def check_the_line():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(2560, 120)
            self.fc2 = nn.Linear(120, 57)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 2560)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    device = torch_utils.select_device('')
    weights = 'D:/TFT_AI_doodoongdeungjang/YOLOv5writtencode/image.pt'
    classes = ('28', '30', '31', '16', '51', '6', '33', '36', '37', '55', '56', '0',
               '2', '1', '38', '44', '45', '49', '50', '52', '9', '10', '11', '14',
               '13', '15', '5', '4', '3', '21', '7', '12', '8', '22', '24', '29', '23',
               '54', '27', '18', '17', '20', '34', '35', '40', '26', '27', '32', '41', '39', '42', '43',
               '47', '46', '48')
    card1 = ImageGrab.grab(bbox=(507, 1010, 682, 1040))
    card2 = ImageGrab.grab(bbox=(697, 1010, 872, 1040))
    card3 = ImageGrab.grab(bbox=(887, 1010, 1062, 1040))
    card4 = ImageGrab.grab(bbox=(1077, 1010, 1252, 1040))
    card5 = ImageGrab.grab(bbox=(1267, 1010, 1442, 1040))

    model = Net()
    checkpoint = torch.load(weights, device)
    model.load_state_dict(checkpoint, device)

    card1 = transforms.ToTensor()(card1).unsqueeze(0)
    card2 = transforms.ToTensor()(card2).unsqueeze(0)
    card3 = transforms.ToTensor()(card3).unsqueeze(0)
    card4 = transforms.ToTensor()(card4).unsqueeze(0)
    card5 = transforms.ToTensor()(card5).unsqueeze(0)

    card1 = model(card1)
    card2 = model(card2)
    card3 = model(card3)
    card4 = model(card4)
    card5 = model(card5)

    card1 = torch.max(card1, 1)
    card2 = torch.max(card2, 1)
    card3 = torch.max(card3, 1)
    card4 = torch.max(card4, 1)
    card5 = torch.max(card5, 1)
    card1 = classes[int(card1[1])]
    card2 = classes[int(card2[1])]
    card3 = classes[int(card3[1])]
    card4 = classes[int(card4[1])]
    card5 = classes[int(card5[1])]

    unify = []
    unify.append(card1)
    unify.append(card2)
    unify.append(card3)
    unify.append(card4)
    unify.append(card5)

    return unify


def check_finished():
    print()


def item():  # 게임 내 박스 파일을 조사하여 캐릭터 배치 현황을 이해하는 함수
    out, source, weights, view_img, save_txt, imgsz = \
        'inference/item_output', 'inference/item_images', 'item.pt', 'store_true', 'store_true', 1920

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = torch_utils.select_device('')
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
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        pred = model(img, augment=None)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

                # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
    print('Done. (%.3fs)' % (time.time() - t0))


def itembox():  # 게임 내 박스 파일을 조사하여 캐릭터 배치 현황을 이해하는 함수
    out, source, weights, view_img, save_txt, imgsz = \
        'inference/block_item_output', 'inference/block_item_images', 'item_box.pt', 'store_true', 'store_true', 1920

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = torch_utils.select_device('')
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
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        pred = model(img, augment=None)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

                # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
    print('Done. (%.3fs)' % (time.time() - t0))


def checktheboxes():  # 게임 내 박스 파일을 조사하여 캐릭터 배치 현황을 이해하는 함수
    out, source, weights, view_img, save_txt, imgsz = \
        'D:/TFT_AI_doodoongdeungjang/YOLOv5writtencode/inference/chamion_output', 'D:/TFT_AI_doodoongdeungjang/YOLOv5writtencode/inference/champion_images', 'D:/TFT_AI_doodoongdeungjang/YOLOv5writtencode/box.pt', 'store_true', 'store_true', 1920

    device = torch_utils.select_device('')
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
    print('Done. (%.3fs)' % (time.time() - t0))


import pandas as pd


def Money_act(stage, number):
    checktheboxes()
    item()
    itembox()
    if stage == 1:
        xp = number
    else:
        xp = 7 * (stage - 1) + number + 4
    level = count_levels(xp, stage)
    champions_file = open('D:/TFT_AI_doodoongdeungjang/YOLOv5writtencode/inference/chamion_output/chamion.txt', 'r')
    items_file = open('D:/TFT_AI_doodoongdeungjang/YOLOv5writtencode/inference/item_output/item.txt', 'r')
    blocked_item_file = open('D:/TFT_AI_doodoongdeungjang/YOLOv5writtencode/inference/block_item_output/item.txt', 'r')
    # current_list = open('D:/TFT_AI_doodoongdeungjang/champion.txt', 'r')
    accompany = open('D:/TFT_AI_doodoongdeungjang/current_level.txt', 'r')
    level_need = pd.read_csv('D:/TFT_AI_doodoongdeungjang/level_needs.csv', header=None)
    gold_need = pd.read_csv('D:/TFT_AI_doodoongdeungjang/gold.csv')
    reroll_need = pd.read_csv('D:/TFT_AI_doodoongdeungjang/reroll_need.csv')
    item_need = pd.read_csv('D:/TFT_AI_doodoongdeungjang/evaluate_item.csv')
    synergy = pd.read_csv('D:/TFT_AI_doodoongdeungjang/synergy.csv')
    batch_item = pd.read_csv('D:/TFT_AI_doodoongdeungjang/batch_item.csv')
    selection = check_the_line()
    if stage == 1:
        stagepoint = number - 1
    else:
        stagepoint = number - 1 + 6 * (stage - 1)
    '''list = current_list.read().split()
    for i in range(len(list)):
        new = list[i]
'''
    chams = []
    synergies = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    job = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    items = []
    needs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        , 0, 0, 0]
    weights = pd.read_csv('D:/TFT_AI_doodoongdeungjang/evaluate_data.csv', header=None)
    # 기록
    while True:
        cham = champions_file.readline().split()
        if not cham: break
        chams.append(cham[0])
        synergies[int(champions[int(cham[0])][1]) - 1] += 1
        job[int(champions[int(cham[0])][2]) - 1] += 1
        job[int(champions[int(cham[0])][3]) - 1] += 1
        item = items_file.readline().split()
        if (int(item[0]) == 10):
            level += 1
        blocked_items = blocked_item_file.readline().split()
        if (int(blocked_items[0]) == 10):
            level += 1
        items.append(item[0])

        items.append(blocked_items[0])
    for i in range(len(chams)):
        needs[int(chams[i]) - 1] += 1
    for i in range(len(selection)):
        Data = int(weights[1][stage]) * (int(needs[int(selection[i])])) + \
               int(weights[4][stage]) * int(selection[i]) + \
               int(synergies[int(champions[int(selection[i])][1]) - 1]) + \
               int(job[int(champions[int(selection[i])][2]) - 1]) + \
               int(job[int(champions[int(selection[i])][3]) - 1])

        if Data > gold_need.values[stagepoint][1]:
            select(buying_batch[i][0], buying_batch[i][1])
            time.sleep(1.0)


# 게임 내부 조작이 아닌 단순 조작에 해당하는 라이브러리

def start_game():  # detect class 에서 화면의 종류를 찾고 객체의 위치를 반환함
    select(855, 845)
    select(855, 845)
    select(855, 845)
    select(855, 845)
    select(855, 845)
    time.sleep(5)
    select(855, 845)
    select(855, 845)
    select(855, 845)
    select(855, 845)
    select(855, 845)

    for i in range(12):
        select(970, 710)
        time.sleep(5)


def finish_game():
    select(840, 520)
    select(840, 520)
    select(840, 520)
    select(840, 520)
    select(840, 520)
    select(840, 520)
    select(840, 520)
    select(840, 520)
    time.sleep(15)
    select(855, 845)
    select(855, 845)
    select(855, 845)
    select(855, 845)
    select(855, 845)
    start_game()


from game import *

if __name__ == '__main__':
    start_game()
    finsihed = False
    time.sleep(5)
    while finsihed != True:
        stage, gauze = check_Stage()
        if (stage == 1 and gauze == 1) or (gauze == 4 and stage != 1):
            Running_ACT(stage, gauze)
        elif (type == 1):
            Money_act(stage, gauze)
            Batch_Act()
            check_GAI()

        finsihed = check_finished()
    finish_game()
