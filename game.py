from Variables import *
from PIL import ImageGrab
from Pi import *
import torch
import pandas as pd
from keys import *
import YOLOv5writtencode.utils.torch_utils as torch_utils
import numpy as np
Epsilon = 0.1




def Batch_Act(stage, gauze):
    stage= int(stage)
    gauze=int(gauze)

    device = torch_utils.select_device('')  # 게임내에서는 cud 0으로 바꾸자!
    targets= pd.read_csv('D:/TFT_AI_doodoongdeungjang/target.csv', header=None)
    champions_file = open('D:/TFT_AI_doodoongdeungjang/YOLOv5writtencode/inference/chamion_output/chamion.txt', 'r')
    batches=[]
    while True:
        cham = champions_file.readline().split()
        if not cham: break
        batches.append([cham[0],cham[1],cham[2]])

    if stage==1:
        target=1
    else:
        target=0

    if target==1:
        saved_model_path = 'win.pt'
    else:
        saved_model_path ='lose.pt'
    model = RCA()
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12'
               ,'13', '14', '15', '16', '17', '18', '19', '20', '21', '22','23','24','25'
               ,'26','27')
    # champions  챔피언의 정보에 해당하는 배치를 두고 계산
    parameter = torch.load(saved_model_path, device)
    model.load_state_dict(parameter, device)
    # 모델을 통과 시켜 무조건 나오는 값을 보고 추론
    if gauze==7:
        level_up()
        level_up()
        level_up()
        level_up()

    for i in range(len(batches)):
        r=np.array([batches[i][0]])

        batch=torch.tensor([int(r)],dtype=torch.float).cuda()
        print(batch)
        print(batch.size())
        result=model(batch)
        result=result.max(dim=0)
        result=classes[int(result[1])]
        select(int(ally_batch[int(result)][0]),int(ally_batch[int(result)][0]))
        take()
        moveon(1920*float(batches[i][0]),1080*float(batches[i][1]),int(ally_batch[int(result)][0]),int(ally_batch[int(result)][0]))

def check_GAI():
    given_image = ImageGrab.grab(bbox=(560, 170, 1360, 720))
    width, height = given_image.size
    pixel_values = list(given_image.getdata())
    if given_image.mode == "RGB":
        channels = 3
    elif given_image.mode == "L":
        channels = 1
    else:
        print("Unknown mode: %s" % given_image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    for y in range(550):
        for x in range(800):
            # print(str(pixel_values[int(x)][int(y)][0])+" "+str(pixel_values[int(x)][int(y)][1])+" "+str(pixel_values[int(x)][int(y)][2])+"\n")

            if (pixel_values[int(x)][int(y)][0] == 255 and pixel_values[int(x)][int(y)][1] == 236 and
                    pixel_values[int(x)][int(y)][2] == 115):
                print(str(x) + "legend " + str(y))
                select(560 + x, 170 + y)
            if (pixel_values[int(x)][int(y)][0] == 250 and pixel_values[int(x)][int(y)][1] == 237 and
                    pixel_values[int(x)][int(y)][2] == 97):
                print(str(x) + "gold " + str(y))
                select(560 + x, 170 + y)
            if (pixel_values[int(x)][int(y)][0] == 1 and pixel_values[int(x)][int(y)][1] == 28 and
                    pixel_values[int(x)][int(y)][2] == 221):
                print(str(x) + "blue " + str(y))
                select(560 + x, 170 + y)
            if (pixel_values[int(x)][int(y)][0] == 87 and pixel_values[int(x)][int(y)][1] == 86 and
                    pixel_values[int(x)][int(y)][2] == 82):
                print(str(x) + "green " + str(y))
                select(560 + x, 170 + y)
