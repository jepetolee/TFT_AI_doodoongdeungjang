import keys as key
import functions as fx
#변수들 도입
start_variables=[1,2,3,4,5,6,7,8]
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
trunks = [1,2,3,4,5,6,7,8,9]#9개의 캐릭터를 담을 수 있는 트렁크

consecutive=0
latest_result =0
#연승을 알기위한 변수들 1은 연승상태, 2는 연패상태 만약 기호가 바뀐다면 consexutive 변수는 조정된다.

rounds=0
survived_enemies =0
survived_allies =0
stage=0
# 다음 변수들은 라운드에 따라 q러닝을 다르게 설계하려고 하기 때문에 중요하다.

lEA=1#level equates accomodates
levupcnt=0#렙업횟수
BARDCNT=0#바드에 의한 랩업 변수, 외계인을 판 함수에 적용되어야 한다.

moneys =0#돈

HP=0#체력을 나타내는 변수로써 성장기 은하계와 꼬꼬마 은하계에 대한 변수를 고려해야한다.

