# 변수들 도입
start_variables = [1, 2, 3, 4, 5, 6, 7, 8]
'''
#시작 변수 표
1. 꼬꼬마 전설이
2. 우주무기고
3. 니코의 세계
4. 성장기 전설이 은하계
5. 초밀도 은하계
6. 교환의 장 
7. 보물창고
8. 성단
위의 내용대로 시작 변수에 따른 게임 기록을 불러오고, 이것에 해당하는 MCTS를 적용할 예정이다.
'''
trunks_x = [460, 560, 660, 760, 860, 960, 1060, 1070, 1080]  # 9개의 캐릭터를 담을 수 있는 트렁크
trunks_y = [760]

consecutive = 0
latest_result = 0
# 연승을 알기위한 변수들 1은 연승상태, 2는 연패상태 만약 기호가 바뀐다면 consexutive 변수는 조정된다.

rounds = 0
survived_enemies = 0
survived_allies = 0
stage = 0
# 다음 변수들은 라운드에 따라 q러닝을 다르게 설계하려고 하기 때문에 중요하다.

lEA = 1  # level equates accomodates
levupcnt = 0  # 렙업횟수
BARDCNT = 0  # 바드에 의한 랩업 변수, 외계인을 판 함수에 적용되어야 한다.

moneys = 0  # 돈

HP = 0  # 체력을 나타내는 변수로써 성장기 은하계와 꼬꼬마 은하계에 대한 변수를 고려해야한다.

# 다음 배치를 대처하는 방식은 고정된 픽셀값마다
# image grab을 쓴 후 cifar100과 같은 방식으로 이미지 인식을 할 것이다.
# 아군의 배열
ally_x = [560, 670, 780, 890, 1000, 1110, 1220, 610, 720, 830, 940, 1050, 1160, 1270]
ally_y = [440, 510, 580, 650]

# 적군의 배열
# 이미지 클레시피케이션으로 대체

'''
--챔피언 체계--
갱플랭크      0 
그레이브즈     1
나르      2
노틸러스 3
녹턴 4
니코 5 
다리우스 6
라칸 7
럼블 8
레오나 9
루시안 10
룰루 11
리븐 12
마스터 이 13
말파이트 14
모데카이저 15
바드 16
바이 17
베인 18 
블리츠크랭크 19
빅토르 20
뽀삐 21
샤코 22
소라카 23
쉔 24 
신 짜오 25
신드라 26
쓰레쉬 27
아리 28     
아우렐리온 솔 29
애니 30
애쉬 31 
야스오 32 
에코 33 
오공 34
우르곳 35
이렐리아 36
이즈리얼 37
일라오이 38
자르반 4세 39
자야 40
잔나 41
제드 42 
제라스 43 
제이스 44
조이 45
직스 46
진 47
징크스 48
카르마 49
카시오페아 50
케이틀린 51
코그모 52
트위스티드 페이트 53
티모 54
피오라 55
피즈 56
'''

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
#  챔피언 마다의 돈, 계열, 직업1, 직업 2 소개
champions = [[5, 6, 7, 13],
             [1, 6, 12, 14],
             [4, 7, 6, 14],
             [2, 7, 3, 14],
             [1, 8, 10, 14],
             [3, 2, 4, 14],
             [2, 6, 1, 14],
             [2, 9, 4, 14],
             [3, 0, 13, 14],
             [1, 3, 3, 14],
             [2, 3, 12, 14],
             [5, 9, 5, 14],
             [4, 4, 0, 14],
             [3, 1, 0, 14],
             [1, 1, 6, 14],
             [2, 5, 3, 14],
             [3, 7, 5, 14],
             [3, 3, 6, 14],
             [3, 3, 11, 14],
             [2, 4, 6, 14],
             [4, 8, 2, 14],
             [1, 2, 3, 14],
             [3, 5, 10, 14],
             [4, 2, 5, 14],
             [2, 4, 0, 14],
             [2, 9, 4, 14],
             [3, 2, 2, 14],
             [5, 4, 1, 14],
             [2, 2, 2, 14],
             [5, 1, 8, 14],
             [2, 0, 2, 14],
             [3, 9, 11, 14],
             [2, 1, 0, 14],
             [5, 3, 10, 14],
             [4, 4, 3, 14],
             [5, 8, 4, 14],
             [4, 3, 0, 1],
             [3, 4, 12, 14],
             [1, 8, 6, 14],
             [1, 5, 4, 14],
             [1, 9, 0, 14],
             [5, 2, 9, 14],
             [2, 1, 10, 14],
             [5, 5, 2, 14],
             [3, 6, 3, 14],
             [1, 2, 2, 14],
             [1, 1, 13, 14],
             [4, 5, 11, 14],
             [4, 1, 12, 14],
             [3, 5, 5, 14],
             [3, 8, 5, 14],
             [1, 4, 11, 14],
             [2, 8, 12, 14],
             [1, 4, 2, 14],
             [4, 7, 11, 14],
             [1, 3, 0, 14],
             [4, 0, 10, 14]]
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
primary_item = [0, 1, 2, 3, 4, 5, 6, 7, 8]

complete_item = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
# 여기서 도적의 장갑같은 변수 고려 필요, 뒤집개 시너지
