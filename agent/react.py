from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.agents import tool
import os
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import PromptTemplate
import ast

load_dotenv()

# Get OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# ChatOpenAI 클래스를 langchain_openai 모듈에서 가져옵니다.
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0, openai_api_key = api_key)

@tool
def recommend_popular_menu(text: str) -> str:
    """인기있는 메뉴를 추천해주는 함수"""
    famous_menu = '아메리카노'
    return famous_menu

@tool
def recommend_menu(text: str) -> str:
    """Semantic search로 메뉴 추천"""
    print('recommend menu')

@tool
def ordering(text: str) -> str:
    """주문이 진행됨, 메타 필터링을 활용"""
    return 'ordering'

@tool
def calculate_price(text: str) -> str:
    """메뉴명과 메뉴 개수를 뽑아와서 정의된 가격을 참고하여, 가격을 출력"""
    total_price = 0
    meta = ast.literal_eval(extract_meta(text))
    print(meta)
    for i in range(len(meta)):
        print(meta[i])
        menu = meta[i]['name']
        n = meta[i]['num']
        if menu == '아메리카노':
            price = 3500
        elif menu == '아이스카페라떼':
            price = 5000
        total_price += price * n
 
    return f"총 합쳐서 {total_price}원입니다."

@tool
def qna(input: str) -> str:
    """카페와 관련된 문의 사항을 응답하는 LLM 함수"""
    response = ChatOpenAI(
        model_name="gpt-4o-mini",openai_api_key = api_key
    )
    messages = [
        (
            "system",
            """너는 지금부터 카페의 매니저로 임명합니다. 매니저는 카페에 들어오는 문의를 모두 응답해야 합니다. 카페에 대한 매뉴얼을 모두 숙지하여 내용에 있는 내용만
            대답을 하고 정보에 담겨있지 않는 내용은 '제가 잘 모르는 정보입니다.'라고 대답하세요.
            매뉴얼에 있는 내용이라면 매뉴얼대로 답변하지 말고 친절하게 설명하듯 말씀해주세요.
            
            [카페 운영 매뉴얼]
            1. 카페의 운영시간은 8:00부터 22:00까지 운영합니다. 카페는 커피와 음료, 베이커리를 제공하고 있으며 실제 있는 메뉴만 판매하고 있습니다. 
            2. 음료나 음식에 대한 환불은 마시지 않았다면 가능합니다. 하지만 손님이 마셨다면 환불은 안됩니다.
            3. 음료 제작 시간은 한 음료당 3분씩 걸립니다. 두 잔은 6분, 세 잔은 9분 순으로 책정됩니다.
            4. 카페의 총 좌석은 30석이며 4인 좌석 5개, 2인 좌석 5개 입니다.
            5. 음료의 가격 할인은 없고 할인을 요구하면 매너있고 친절하게 안된다고 응대하세요.
            6. 가맹 문의는 받지 않습니다.
            7. 음료를 시키면 카페 이용시간은 대체로 2시간입니다. 하지만 덜 있어도 되고 더 있어도 됩니다.
            
            위의 내용을 참고하고 매뉴얼에 없는 내용은 매니저로서 응답할 수 있으면 하고, 매니저가 답변할 수 없는 질문은 '저도 잘 모르겠습니다.'라고 답변하세요.
            """,
        ),
        (
            "user",
            input
            ),
    ]
    return response.invoke(messages).content

def extract_meta(text: str) -> list:
    response = ChatOpenAI(
        model_name="gpt-4o-mini",openai_api_key = api_key
    )
    messages=[
        (
            "system",
            """너는 사용자의 입력을 보고 주문서를 작성해야 하는 경력있는 종업원이야. 너는 카페의 전문가고 사용자의 입력에서 알 수 었는 정보는 모두 'None'으로 처리해.
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
        ),
        (
            "user",
            text,
        )
    ]
    return response.invoke(messages).content


tools = [recommend_popular_menu, recommend_menu, ordering, qna, calculate_price]


tool_names = ", ".join([tool.name for tool in tools])

template = """
    Answer the following questions as best you can. You are a friendly café staff member. 
    Analyze the customer's request to understand their intent and respond kindly. 
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question. PLEASE ANSWER in KOREAN. 

    Begin!
    

    Question: {input}
    Thought: {agent_scratchpad}
    """
prompt = PromptTemplate(input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template=template
)
# prompt에 tools와 tool_names 값을 설정
prompt = prompt.partial(
    tools="\n".join([f"- {tool.name}" for tool in tools]),
    tool_names=tool_names
)


agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "환불 어떻게 해요! 당장 환불해줘."})

print(f'답변: {response["output"]}')