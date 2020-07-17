import pyautogui
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

def  sell():
    pyautogui.press('e')

def cost_up():
    pyautogui.press('f')

def move(x,y):
    pyautogui.click(x,y,button='right')

def select(x,y):
    pyautogui.click(x,y,button='left')

def lock():
    pyautogui.click(1500,900,button='left')

def check_Enemies():
    pyautogui.press('q')
