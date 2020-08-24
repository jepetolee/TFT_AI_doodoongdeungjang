from keys import *
from operate import *
from functions import *
from Variables import *
import numpy as np
from PIL import ImageGrab
from Pi import *
import torch
from YOLOv5writtencode.finding import *
import torch.backends.cudnn as cudnn


def Running_ACT(x, y, saved_batches):  # 회전 초밥 한정
    round = rounds
    device = torch.device('')  # 게임내에서는 cud 0으로 바꾸자!
    saved_model_path = 'running_act_for_cycle.pt'  # x,y 좌표에 해당하는 모델 데려오고, 이동
    model = torch.load(weights=saved_model_path, map_location=device)
    target = []  # 선택 챔피언 순서,MCTS를 불러와서 이제 값을 구할 거임

    check = 1  # look(saved_batches, target)
    if check == 1:
        point = 10
    elif check == 2:
        point = 5
    elif check == 3:
        point = -5
    else:
        point = -10


def Batch_Act(batches):
    device = torch.device('')  # 게임내에서는 cud 0으로 바꾸자!
    saved_model_path = 'batch_act.pt'
    # champions  챔피언의 정보에 해당하는 배치를 두고 계산
    model = torch.load(weights=saved_model_path, map_location=device)
    # 모델을 통과 시켜 무조건 나오는 값을 보고 추론
    result_batches = []
    for x1, y1, x2, y2 in zip(batches[0], batches[1], result_batches[0], result_batches[1]):
        moveon(x1, y1, x2, y2)


# reward= Wins_to_reward()# 몬테 카를로 식으로 구성되어있는 wins와 false 데이터 이를 통해 결과를 산출해 낸다.


def Money_act(money, wins, stage_on_game):
    if stage_on_game != 1 and wins == 1:
        print(money)
    device = torch.device('')  # 게임내에서는 cud 0으로 바꾸자!
    saved_model_path = 'money_act.pt'
    model = torch.load(weights=saved_model_path, map_location=device)




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
