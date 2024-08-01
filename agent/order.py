from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI()

def purpose(input):
    response = client.chat.completions.create(
        messages=[
            {
                "role" : "system",
                "content" : """너는 사용자의 입력을 보고 주문서를 작성해야 하는 경력있는 종업원이야. 너는 카페의 전문가고 사용자의 입력에서 알 수 었는 정보는 모두 'None'으로 처리해.
                사용자의 주문에서 너가 찾아야 하는 정보는 총 네 가지야. 1. 사이즈 2. 온도 3. 카페인 여부 4. 메뉴 이름 5. 개수
                다음과 같은 예시를 보고 출력해줘. 대신 줄임말이 있을 수 있으니 '아아'는 '아이스 아메리카노'의 줄임말이고, '아바라'는 '아이스 바닐라 라떼'의 줄임말이고, '아샷추'는 '아이스티에 샷 추가'야. 줄임말을 추측해줘.
                
                name : 음료 종류
                category : '커피', '음료', '베이커리'
                temporary : 'iced', 'hot', None
                has_caffeine : 1, 0
                size : 'R', 'MD', 'EX', 'L', '10T', '20T', '30T', '100T', '80+20T', None
                num : 개수, None
                
                입력 예시 : 아이스 아메리카노 디카페인 3개
                출력 예시 : [{'name':'아메리카노', 'category':'커피', 'temporary':'iced', 'has_caffeine':1, 'size':None, 'num':3}]
                
                입력 예시 : 아아 1잔과 따뜻한 카페라떼 2잔
                출력 예시 : [{'name':'아메리카노', 'category':'커피', 'temporary':'iced', 'has_caffeine':1, 'size':None, 'num':1},{'name':'카페 라떼', 'category':'커피', 'temporary':'hot', 'has_caffeine':1, 'size':None, 'num':2}]
                
                입력 예시 : 자몽에이드 M 사이즈 2개랑 따뜻한 모카라떼 1잔, 아바라 1잔
                출력 예시 : [{'name':'자몽에이드', 'category':'음료', 'temporary':'iced', 'has_caffeine':0, 'size':'R', 'num':2},{'name':'모카 라떼', 'category':'커피', 'temporary':'hot', 'has_caffeine':1, 'size':None, 'num':1}, {'name':'바닐라 라떼', 'category':'커피', 'temporary':'iced', 'has_caffeine':1, 'size':None, 'num':1}]"""
            },
            {
                "role": "user",
                "content": input,
            }
        ],
        model="gpt-4o-mini",
    )
    return response.choices[0].message.content

print(purpose("딸기라떼 L 사이즈 1개와 아샷추 1개"))