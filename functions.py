#데미지를 세주는 함수
def count_damage(unit,stage):
    unit_damage =[2,4,6,8,10,11,12,13,14,15]
    stages = [0,0,1,2,5,10,15]
    if(stage>7):
        stage=7
    return unit_damage[unit-1]+stages[stage-1]

#위 함수들은 칸을 줄일 수 있는 방법에 대한 여지를 준다.
def left_for_2(get):
    return 3-get

def left_for_3(get):
    return 9 -get

#라운드에 따라 변수 계산을 따로 해줘야 하기 때문에 만든 함수
'''
0. 크립 라운드
1. 초밥집
2. 일반 전투
'''
def counting_map(round):
    if(round==1):
        return 1
    if(1<round and round<=4):
        return 0
    if((round-4)%7==4):
        return 1
    else:
        return 2

#라운드에 따라 스테이지가 어느 정도인지 계산해 주는 함수
def counting_stage(round):
    if(round<=4):
        return 1
    else:
        return int((round-4)/7)+2

#연승 연패 값을 계산해주는 함수
def lasting_winsNloss(count):
    if (count==1 or count==-1):
        return 0
    elif (count==3 or count==-3 or count==2 or count==-2):
        return 1
    elif (count==4 or count==-4):
        return 2
    else:
        return 3

#이자를 세는 함수
def counting_interest( golds,consecute,round):
    interest_A = int(golds/10)
    if(interest_A>5):
        interest_A=5
    interest_B = lasting_winsNloss(consecute)
    if (round==1):
        interest_C=0
    elif(round>1 and round<=3):
        interest_C =2
    elif(round==4):
        interest_C =3
    elif(round==5):
        interest_C=4
    else:
        interest_C=5
    return interest_A+interest_B+interest_C

#렙업을 위한 변수 제작
def lev_needs(levupcnt,rounds,BARDcnt):
    lev_A=2*(rounds-1)
    lev_B= levupcnt*4
    lev_C=BARDcnt
    return lev_A+lev_B+lev_C
