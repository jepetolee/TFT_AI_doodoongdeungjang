import torch.backends.cudnn as cudnn
import argparse
from YOLOv5writtencode.models.experimental import *
from YOLOv5writtencode.utils.datasets import *
from YOLOv5writtencode.utils.utils import *


def findthescreen():  # 게임 등수, 회전초밥, 게임 재시작 화면 1, 게임 재시작 화면 2, 게임 선택창 ,일반
    screen = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 1080)))
    model = tf.keras.models.load_model('C:/Users/user/PycharmProjects/BRAIN.net/YOLOv5writtencode/model/save.h5')
    image_array = np.asarray(screen)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction


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


def checktheboxes():  # 게임 내 박스 파일을 조사하여 캐릭터 배치 현황을 이해하는 함수
    out, source, weights, view_img, save_txt, imgsz = \
        'inference/output', 'inference/images', 'box.pt', 'store_true', 'store_true', 1920

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


checktheboxes()
