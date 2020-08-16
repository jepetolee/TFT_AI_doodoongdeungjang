from keys import *
from YOLOv5writtencode.model import *
from operate import *
from functions import *
from Pi import*
import torch
import torch.backends.cudnn as cudnn

def Running_ACT(x,y):# 회전 초밥 한정
    device = torch.device('cpu')  # 게임내에서는 cud 0으로 바꾸자!
    saved_model_path = 'running_act.pt'
    model = torch.load(weights=saved_model_path, map_location=device)
    # 모델을 데려와서 이제 먹었는지 체크, 그리고 먹었다면, +1, 못먹었다면 -1

def Batch_Act(batches):
    device = torch.device('cpu')  # 게임내에서는 cud 0으로 바꾸자!
    saved_model_path = 'running_act.pt'
    model = torch.load(weights=saved_model_path, map_location=device)
    # 모델을 데려와서 이제 먹었는지 체크, 그리고 먹었다면, +1, 못먹었다면 -1


