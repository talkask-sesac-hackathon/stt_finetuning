import ast
import os
import sys
import json
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent, tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from recommendation import *
from data.vectordb_manager import *


load_dotenv(override=True)
prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
qna_prompt_path = os.path.join(prompt_path, "qna_prompts.txt")
extract_meta_prompt_path = os.path.join(prompt_path, "extract_meta_prompts.txt")
extract_is_popular_prompt_path = os.path.join(prompt_path, "extract_is_popular_prompts.txt")
with open(qna_prompt_path, "r") as f:
    qna_prompt = f.read()
with open(extract_meta_prompt_path, "r") as f:
    extract_meta_prompt = f.read()
with open(extract_is_popular_prompt_path, "r") as f:
    extract_is_popular_prompt = f.read()

llm = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)

@tool
def recommend_popular_menu(user_input: str) -> str:
    """인기있는 메뉴를 추천해주는 함수"""
    messages = [
        ("system", extract_is_popular_prompt),
        ("user", user_input)
    ]
    is_popular = llm.invoke(messages).content
    if is_popular == "예":
        chroma = read_chroma(chroma_path, embedding)
        results = chroma.similarity_search("인기 추천", k=10, filter={"is_popular": True})
        famous_menu = list(set(get_names(results)))
    elif is_popular == "아니요":
        famous_menu = ''
    elif is_popular != "예" and is_popular != "아니요":
        if '인기' in user_input or '추천' in user_input:
            chroma = read_chroma(chroma_path, embedding)
            results = chroma.similarity_search("인기 추천", k=10, filter={"is_popular": True})
            famous_menu = list(set(get_names(results)))
    return famous_menu

@tool
def recommend_menu(user_input: str) -> str:
    """Semantic search로 메뉴 추천"""
    print('recommend menu')

@tool
def ordering(user_input: str) -> str:
    """주문이 진행됨, 메타 필터링을 활용"""
    return 'ordering'

@tool
def qna(user_input: str) -> str:
    """카페와 관련된 문의 사항을 응답하는 LLM 함수"""
    messages = [
        ("system", qna_prompt),
        ("user", user_input),
    ]
    return llm.invoke(messages).content

def load_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = file.read()
        data = json.loads(json_data)
        return data

def find_most_similar_name_with_gpt(target_name, data):
    names = []
    for i in data:
        names.append(i['name'])
    messages = [
        (
            "system",
            f"""당신의 임무는 사용자의 입력과 같거나 비슷한 이름을 하나 출력하는 겁니다. 예를 들어 사용자가 "라떼" 라고 말하면 아래에 주어지는 데이터에서
이름을 하나 뽑아내주세요. 비슷한 이름이 없어도 최대한 유사도가 비슷한 이름을 선택하세요.

{names}
            """,
        ),
        (
            "human",
            target_name
            ),
    ]
    
    closest_match = llm.invoke(messages).content
    return closest_match

def find_item_by_name(data, target_name):
    for item in data:
        if item.get('name') == target_name:
            return item

def check_menu_order(input: str, menu: str, order: str):
    messages = [
        (
            "system",
            f"""당신은 매뉴판의 내용과 주문 내용을 보고 필요한 정보를 다 입력받았으면 '잠시만 기다려 주세요'를 출력하고, 만일 아직 정보가 필요하다면 랜덤으로
필요한 정보를 사용자에게 물어보는 질문을 제작해 주세요.
{menu}

사용자의 입력은 다음과 같습니다.
{order}
"""
        ),
        ("human", input)
    ]
    
    closest_match = llm.invoke(messages).content
    return closest_match

def compare_and_fill_meta(menu: dict, actual: dict) -> dict:
    """필터링된 결과와 실제 메타 정보를 비교하고 부족한 정보를 채우는 함수"""
    messages = [
        (
            "system",
            f"""다음의 메뉴 정보와 실제 주문 정보를 확인하고 음료를 만들기 위해 아직 주문받지 못한 내용을 모두 사용자에게 제공해달라고 요청하세요
영어로 되어있다면 한글로 바꿔서 사용자에게 요청하세요, 메뉴 이름은 메뉴 정보의 내용을 참조하지 말고 사용자가 입력한 현재 주문의 내용으로 사용하세요.

필요한 메뉴 정보
{menu}

현재 주문 내역
{actual}

예를 들어, 아이스 아메리카노를 만들 때 종류, 온도, 이름이 필요한데 사용자가 이름과 종류만 말해줬다면 '주문하시는 음료의 온도를 말씀해주세요' 라고 답변하세요.
"""
        ),
        ("user", "무엇을 더 주문해야 할까요?")
    ]
    answer = llm.invoke(messages).content
    return answer

def final_order_check(first: dict, second: dict) -> dict:
    """필터링된 결과와 실제 메타 정보를 비교하고 부족한 정보를 채우는 함수"""
    response = ChatOpenAI(
        model_name="gpt-4o-mini",
    )
    messages = [
        (
            "system",
            f"""첫번째 주문 내역과 두번째 주문 내역을 확인해서 사용자가 음료를 어떻게 주문했는지 최종 확인하는 메세지를 출력해주세요.

필요한 메뉴 정보
{first}

현재 주문 내역
{second}

대신 없는 정보다 None과 같은 정보가 있다면 생성하지 말고 없는 상태로 주문을 확인하세요.
"""
        ),
        ("user", "주문 확인해주세요.")
    ]
    answer = response.invoke(messages).content
    return answer

@tool
def compare_order_menu(user_input:str) -> str:
    """사용자가 주문을 하면 메뉴에 있는 음료 이름과 비교해서 필요한 정보가 빠진 것이 없는지 확인하는 함수"""
    first_need = ast.literal_eval(extract_meta(user_input))

    file_path = os.path.join(prompt_path, "ediya_menu_collection.json")
    data = load_json_from_file(file_path)

    target_name = user_input
    # 갖고 있는 메뉴 중 비슷한 메뉴 이름
    most_similar = find_most_similar_name_with_gpt(target_name, data)
    
    info = find_item_by_name(data, most_similar)
    more_need = []
    for i in first_need:
        more_need.append(compare_and_fill_meta(info, i))
    ment = ''.join(more_need)
    plus_order = input(ment)
    second_need = extract_meta(plus_order)
    print(final_order_check(first_need, second_need))
        
    return '주문을 완료했습니다. 잠시만 기다려주세요.'

def extract_meta(text: str) -> list:
    messages=[
        ("system", extract_meta_prompt),
        ("user", text)
    ]
    return llm.invoke(messages).content

 
tools = [recommend_popular_menu, recommend_menu, ordering, qna, compare_order_menu]


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

response = agent_executor.invoke({"input": "카라멜 마끼아또 주세요"})

print(f'답변: {response["output"]}')