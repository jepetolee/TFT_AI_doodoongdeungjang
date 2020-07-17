# 데미지를 세주는 함수
def count_damage(unit, stage):
    unit_damage = [2, 4, 6, 8, 10, 11, 12, 13, 14, 15]
    stages = [0, 0, 1, 2, 5, 10, 15]
    if stage > 7:
        stage = 7
    return unit_damage[unit - 1] + stages[stage - 1]


# 위 함수들은 칸을 줄일 수 있는 방법에 대한 여지를 준다.
def left_for_2(get):
    return 3 - get


def left_for_3(get):
    return 9 - get


# 라운드에 따라 변수 계산을 따로 해줘야 하기 때문에 만든 함수
'''
0. 크립 라운드
1. 초밥집
2. 일반 전투
'''


def counting_map(round):
    if round == 1:
        return 1
    if 1 < round <= 4:
        return 0
    if (round - 4) % 7 == 4:
        return 1
    else:
        return 2


# 라운드에 따라 스테이지가 어느 정도인지 계산해 주는 함수
def counting_stage(round):
    if round <= 4:
        return 1
    else:
        return int((round - 4) / 7) + 2


# 연승 연패 값을 계산해주는 함수
def lasting_winsNloss(count):
    if count == 1 or count == -1:
        return 0
    elif count == 3 or count == -3 or count == 2 or count == -2:
        return 1
    elif count == 4 or count == -4:
        return 2
    else:
        return 3


# 이자를 세는 함수
def counting_interest(golds, consecute, round):
    interest_A = int(golds / 10)
    if interest_A > 5:
        interest_A = 5
    interest_B = lasting_winsNloss(consecute)
    if round == 1:
        interest_C = 0
    elif 1 < round <= 3:
        interest_C = 2
    elif round == 4:
        interest_C = 3
    elif round == 5:
        interest_C = 4
    else:
        interest_C = 5
    return interest_A + interest_B + interest_C


# 렙업을 위한 변수 제작
def lev_needs(levupcnt, rounds, BARDcnt):
    lev_A = 2 * (rounds - 1)
    lev_B = levupcnt * 4
    lev_C = BARDcnt
    return lev_A + lev_B + lev_C


# 단 포오네 변수를 고려해 줄것
def count_levels(xp, round):
    if round == 1:
        return 1
    elif 0 <= xp < 2:
        return 2
    elif 2 <= xp < 8:
        return 3
    elif 8 <= xp < 18:
        return 4
    elif 18 <= xp < 38:
        return 5
    elif 38 <= xp < 70:
        return 6
    elif 70 <= xp < 120:
        return 7
    elif xp >= 120 and 186:
        return 8
    elif xp >= 186:
        return 9


# 다음은 시너지 제작을 위한 시너지 계수 카운팅 알고리즘이다.

'''
--시너지 체계--
메카 파일럿 0
반군 1
별수호자 2
사이버 네틱 3
시공간 4
암흑의 별 5
우주해적 6
우주 비행사 7
전투기계 8
천상 9
'''


def check_origin(origin_code, counts):
    if origin_code == 0:
        return int(counts / 3)

    elif origin_code == 1:
        return int(counts / 3)

    elif origin_code == 2:
        return int(counts / 3)

    elif origin_code == 3:
        return int(counts / 3)

    elif origin_code == 4:
        return int(counts / 2)

    elif origin_code == 5:
        return int(counts / 2)

    elif origin_code == 6:
        return int(counts / 2)

    elif origin_code == 7:
        return int(counts / 3)

    elif origin_code == 8:
        return int(counts / 2)

    elif origin_code == 9:
        return int(counts / 2)


'''
--직업 체계--
검사 0
마나약탈자 1
마법사 2 
선봉대 3
수호자 4
신비술사 5
싸움꾼 6
용병 7
우주선 8
인도자 9
잠입자 10
저격수 11
총잡이 12
폭파광 13
NULL 14
'''


def check_class(classes, counts):
    if classes == 0:
        return int(counts / 2)
    elif classes == 1:
        return int(counts / 2)
    elif classes == 2:
        return int(counts / 2)
    elif classes == 3:
        return int(counts / 2)
    elif classes == 4:
        return int(counts / 2)
    elif classes == 5:
        return int(counts / 2)
    elif classes == 6:
        return int(counts / 2)
    elif classes == 7:
        return counts
    elif classes == 8:
        return counts
    elif classes == 9:
        return counts
    elif classes == 10:
        return int(counts / 2)
    elif classes == 11:
        return int(counts / 2)
    elif classes == 12:
        return int(counts / 2)
    elif classes == 13:
        return int(counts / 3)
    elif classes == 14:
        return 0


# 위는 아이템의 조합을 계산해주는 함수이다.
'''
--기초 아이템-- 
BF 대검 0
곡궁 1
쇠사슬 조끼 2
음전자 망토 3
쓸데없이 큰 지팡이 4
여신의 눈물 5
거인의 허리띠 6
뒤집개 7
연습용 장갑 8
'''

'''
--핵심 아이템--
죽음의 검 0
거인학살자 1
수호천사 2
피바라기 3
마법공학 총검 4
쇼진의 창 5
지크의 전령 6 
몰락한 왕의검 7
무한의 대검 8
고속연사포 9
거인의 결의 10
루난의 허리케인 11
구인수의 격노검 12 
스태틱의 단검 13
즈롯 차원문 14
잠입자의 발톱 15
최후의 속삭임 16
덤불조끼 17
파쇄검 18
강철의 솔라리 팬던트 19 
얼어붙은 심장 20
붉은 덩굴 정령 21
반군의 메달 22
침묵의 장막 23
용의 발톱 24
이온 충격기 25
힘의 성배 26
서풍 27
천상의 구 28
수은 29
라바돈의 죽음의 모자 30
루덴의 메아리 31
모렐로노미콘 32
전투기계 방패 33
보석 건틀릿 34
푸른 파수꾼 35
구원 36
별 수호자 37
정의의 손길 38
워모그의 갑옷 39
수호자의 흉갑 40
덫 발톱 41
대자연의 힘 42
암흑의 별 심장 43
도적의 장갑 44
'''


def item_combine(A, B):
    if A == 0 and B == 0:
        return 0
    elif (A == 0 and B == 1) or (A == 1 and B == 0):
        return 1
    elif (A == 0 and B == 2) or (A == 2 and B == 0):
        return 2
    elif (A == 0 and B == 3) or (A == 3 and B == 0):
        return 3
    elif (A == 0 and B == 4) or (A == 4 and B == 0):
        return 4
    elif (A == 0 and B == 5) or (A == 5 and B == 0):
        return 5
    elif (A == 0 and B == 6) or (A == 6 and B == 0):
        return 6
    elif (A == 0 and B == 7) or (A == 7 and B == 0):
        return 7
    elif (A == 0 and B == 8) or (A == 8 and B == 0):
        return 8
    elif A == 1 and B == 1:
        return 9
    elif (A == 1 and B == 2) or (A == 2 and B == 1):
        return 10
    elif (A == 1 and B == 3) or (A == 3 and B == 1):
        return 11
    elif (A == 1 and B == 4) or (A == 4 and B == 1):
        return 12
    elif (A == 1 and B == 5) or (A == 5 and B == 1):
        return 13
    elif (A == 1 and B == 6) or (A == 6 and B == 1):
        return 14
    elif (A == 1 and B == 7) or (A == 7 and B == 1):
        return 15
    elif (A == 1 and B == 8) or (A == 8 and B == 1):
        return 16
    elif A == 2 and B == 2:
        return 17
    elif (A == 2 and B == 3) or (A == 3 and B == 2):
        return 18
    elif (A == 2 and B == 4) or (A == 4 and B == 2):
        return 19
    elif (A == 2 and B == 5) or (A == 5 and B == 2):
        return 20
    elif (A == 2 and B == 6) or (A == 6 and B == 2):
        return 21
    elif (A == 2 and B == 7) or (A == 7 and B == 2):
        return 22
    elif (A == 2 and B == 8) or (A == 8 and B == 2):
        return 23
    elif A == 3 and B == 3:
        return 24
    elif (A == 3 and B == 4) or (A == 4 and B == 3):
        return 25
    elif (A == 3 and B == 5) or (A == 5 and B == 3):
        return 26
    elif (A == 3 and B == 6) or (A == 6 and B == 3):
        return 27
    elif (A == 3 and B == 7) or (A == 7 and B == 3):
        return 28
    elif (A == 3 and B == 8) or (A == 8 and B == 3):
        return 29
    elif A == 4 and B == 4:
        return 30
    elif (A == 4 and B == 5) or (A == 5 and B == 4):
        return 31
    elif (A == 4 and B == 6) or (A == 6 and B == 4):
        return 32
    elif (A == 4 and B == 7) or (A == 7 and B == 4):
        return 33
    elif (A == 4 and B == 8) or (A == 8 and B == 4):
        return 34
    elif A == 5 and B == 5:
        return 35
    elif (A == 5 and B == 6) or (A == 6 and B == 5):
        return 36
    elif (A == 5 and B == 7) or (A == 7 and B == 5):
        return 37
    elif (A == 5 and B == 8) or (A == 8 and B == 5):
        return 38
    elif A == 6 and B == 6:
        return 39
    elif (A == 6 and B == 7) or (A == 7 and B == 6):
        return 40
    elif (A == 6 and B == 8) or (A == 8 and B == 6):
        return 41
    elif A == 7 and B == 7:
        return 42
    elif (A == 7 and B == 8) or (A == 8 and B == 7):
        return 43
    elif A == 8 and B == 8:
        return 44
