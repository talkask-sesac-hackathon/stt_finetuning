import os
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)
openai = ChatOpenAI(model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"))

menu_recommendation_prompt_path = os.path.join(os.path.dirname(__file__), "data", "menu_recommendation_prompts.txt")
with open(menu_recommendation_prompt_path, "r") as f:
    menu_recommendation_prompt = f.read()

chat_prompt = ChatPromptTemplate([
    ("system", menu_recommendation_prompt + "{document_input}"),
    ("human", "{user_input}")
])

chain = chat_prompt | openai

def get_menu_recommendation(document_input: str, user_input: str) -> str:
    return chain.invoke({"document_input": document_input, "user_input": user_input}).content

def delete_pricing_from_documents(documents: list[dict]) -> list[dict]:
    for item in documents:
        del item["pricing"]
    return documents

def dict_to_tuple_string(dictionary: dict) -> str:
    return str(dictionary).replace("{", "(").replace("}", ")")

if __name__ == "__main__":
    fake_retrieved_json_data = """
[
    {
        "name": "HOT 아메리카노",
        "category": "커피",
        "description": "가장 인기있고 기본적인 따뜻한 블랙 커피",
        "pricing": {
            "L": 3200,
            "EX": 4200
        },
        "is_popular": true,
        "has_caffeine": true,
        "temperature": "hot"
    },
    {
        "name": "ICED 아메리카노",
        "category": "커피",
        "description": "가장 인기있고 기본적인 시원한 블랙 커피",
        "pricing": {
            "L": 3200,
            "EX": 4200
        },
        "is_popular": true,
        "has_caffeine": true,
        "temperature": "iced"
    }
]
"""
    def show_chat(user_input: str, chat_responce: str):
        print("사용자 입력:", user_input)
        print("챗봇 응답:", chat_responce)
        print()

    fake_retrieved_documents = json.loads(fake_retrieved_json_data)
    fake_retrieved_documents = delete_pricing_from_documents(fake_retrieved_documents)
    document_input = dict_to_tuple_string(fake_retrieved_documents)

    user_input = "따뜻하고 카페인이 없는 음료 추천해줘"
    chat_responce = get_menu_recommendation(document_input, user_input)
    show_chat(user_input, chat_responce)

    user_input = "시원하고 카페인이 있는 음료 추천해줘"
    chat_responce = get_menu_recommendation(document_input, user_input)
    show_chat(user_input, chat_responce)