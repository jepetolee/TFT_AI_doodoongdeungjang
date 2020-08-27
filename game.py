from keys import *
from operate import *
from functions import *
from Variables import *
import numpy as np
from PIL import ImageGrab
from Pi import *
import torch
import YOLOv5writtencode.finding as find
import torch.backends.cudnn as cudnn
import pandas as pd


Epsilon = 0.1

def Running_ACT(stage, grade):  # Epsilon- Episode greedy algorithm

    if stage == 1:
        champions_file = open('YOLOv5writtencode/inference/chamion_output/champion.txt', 'r')
        items_file = open('YOLOv5writtencode/inference/item_output/item.txt', 'r')
        weights = pd.read_csv('evaluate_data.csv', header=None)
        item_weights = pd.read_csv('evaluate_item.csv', header=None)
        synergy_fitness = [0, 0, 0, 0, 0, 0, 0, 0]  # 시너지 필요도 계산 나중에 불러와야합니다. 남은 시너지,
        selection = []
        item_needs=[]
        item_recurrent=open('./item_recurrent.txt', 'r')
        count = 0
        while True:
            cham = champions_file.readline().split()
            item = items_file.readline().split()
            recurrent_item=item_recurrent.readline().split()
            if not cham: break
            selection.append([cham[0], item[0]])
            count += 1


        for i in range(count):
            Data = [int(weights[2][stage]) * (int(item_needs[0])) + \
                    int(weights[1][stage]) * (int(synergy_fitness[0])) + \
                    int(weights[4][stage]) * int(selection[i][0]) + \
                    int(weights[3][int(stage)]) * int(champions[int(selection[i][0])][0])]
            selection[i].append(Data[0])
            i += 1
        selection.sort(key=lambda x: x[2], reverse=True)
        print(selection)  # 이후에 예측거리만을 클릭한다.

    else:
        # 등수에 따른 경과시간을 잰후

        champions_file = open('YOLOv5writtencode/inference/chamion_output/champion.txt', 'r')
        items_file = open('YOLOv5writtencode/inference/item_output/item.txt', 'r')
        weights = pd.read_csv('evaluate_data.csv', header=None)
        item_weights = pd.read_csv('evaluate_item.csv', header=None)
        synergy_fitness = [0, 0, 0, 0, 0, 0, 0, 0]  # 시너지 필요도 계산 나중에 불러와야합니다. 남은 시너지,
        item_needs = [0]  # 아이템 필요도를 산출하는 형식의 데이터도 구현이 필요합니다. 필요한개수, 그리고, 아이템 번호의 강화학습 정도
        selection = []
        count = 0
        while True:
            cham = champions_file.readline().split()
            item = items_file.readline().split()
            if not cham: break
            selection.append([cham[0], item[0]])
            count += 1

        for i in range(count):
            Data = [int(weights[2][stage]) * (int(item_needs[0])) + \
                    int(weights[1][stage]) * (int(synergy_fitness[0])) + \
                    int(weights[4][stage]) * int(selection[i][0]) + \
                    int(weights[3][int(stage)]) * int(champions[int(selection[i][0])][0])]
            selection[i].append(Data[0])
            i += 1

        selection.sort(key=lambda x: x[2], reverse=True)
        print(selection)


def Batch_Act(batches):
    device = torch.device('')  # 게임내에서는 cud 0으로 바꾸자!
    saved_model_path = 'batch.pt'
    # champions  챔피언의 정보에 해당하는 배치를 두고 계산
    model = torch.load(weights=saved_model_path, map_location=device)
    # 모델을 통과 시켜 무조건 나오는 값을 보고 추론
    result_batches = []
    for x1, y1, x2, y2 in zip(batches[0], batches[1], result_batches[0], result_batches[1]):
        moveon(x1, y1, x2, y2)


# reward= Wins_to_reward()# 몬테 카를로 식으로 구성되어있는 wins와 false 데이터 이를 통해 결과를 산출해 낸다.


def Money_act(money, wins, stage,number):
    champions_file = open('YOLOv5writtencode/inference/chamion_output/champion.txt', 'r')
    items_file = open('YOLOv5writtencode/inference/item_output/item.txt', 'r')
    blocked_item_file=open('YOLOv5writtencode/inference/block_item_output/item.txt', 'r')
    current_list= open('champion.txt','r')
    list=current_list.read().split()
    for i in range(len(list)):
        new=list[i]

    while True:
        cham = champions_file.readline().split()
        if not cham: break
        synergies = []
        job=[]
        items=[]
        synergies.append(champions[int(cham[0])][1])
        job.append([champions[int(cham[0])][2],champions[int(cham[0])][3]])
        item = items_file.readline().split()
        blocked_items= blocked_item_file.readline().split()
        items.append(item[0])
        items.append(blocked_items[0])
    print(list[0])





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

Money_act(50,30,20,3)