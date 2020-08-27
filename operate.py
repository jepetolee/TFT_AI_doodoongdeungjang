from keys import *
from keys import *
from operate import *
from functions import *
from Variables import *
from game import *
import numpy as np
from PIL import ImageGrab
from Pi import *
import torch
import YOLOv5writtencode.finding as find
import torch.backends.cudnn as cudnn
import pandas as pd


# 게임 내부 조작이 아닌 단순 조작에 해당하는 라이브러리

def start_game():  # detect class 에서 화면의 종류를 찾고 객체의 위치를 반환함
    select(855, 845)
    select(855, 845)
    select(855, 845)
    select(855, 845)
    select(855, 845)

    select(359, 845)
    select(360, 845)
    select(358, 845)
    select(361, 845)
    select(357,845)
    while(1):
        given_image = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
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

        if (pixel_values[960][710][0] == 92 and pixel_values[960][710][1] == 91 and
                        pixel_values[960][710][2] == 87):break
        select(970, 710)


if __name__ == '__main__':
    start_game()





