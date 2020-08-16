import pyautogui
import time

'''
w 배치, 대기석 복귀
e 판매 
d 리롤
f 레벨업
q 상대방 보기
'''


def put():
    pyautogui.press('w')


def take():
    pyautogui.press('w')


def cylcle():
    pyautogui.press('d')


def sell():
    pyautogui.press('e')


def cost_up():
    pyautogui.press('f')


def moveon(x1, y1, x2, y2):
    pyautogui.mouseDown(x1, y1, button='left')
    time.sleep(0.2)
    pyautogui.mouseUp(x2, y2, button='left')


def select(x, y):
    pyautogui.click(x, y, button='left')


def move(x, y):
    pyautogui.click(x, y, button='right')
    result=0
    return result


def lock():
    pyautogui.click(1500, 900, button='left')


def check_Enemies():
    pyautogui.press('q')
