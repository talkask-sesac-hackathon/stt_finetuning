import ast
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent, tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)
prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
qna_prompt_path = os.path.join(prompt_path, "qna_prompts.txt")
extract_meta_prompt_path = os.path.join(prompt_path, "extract_meta_prompts.txt")
with open(qna_prompt_path, "r") as f:
    qna_prompt = f.read()
with open(extract_meta_prompt_path, "r") as f:
    extract_meta_prompt = f.read()

llm = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)

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
    messages = [
        ("system", qna_prompt),
        ("user", input),
    ]
    return llm.invoke(messages).content

def extract_meta(text: str) -> list:
    messages=[
        ("system", extract_meta_prompt),
        ("user", text)
    ]
    return llm.invoke(messages).content

 
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