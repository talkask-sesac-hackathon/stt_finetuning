from pathlib import Path
from typing import List, Dict
import json

from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document


def read_json(path: str | Path) -> List[Dict[str, str | bool | dict[str, int]]]:
    """ JSON 데이터 구조
    [
        {
            "name": "메뉴 이름",       # str
            "category": "음료 종류",   # str
            "description": "설명",    # str
            "pricing": {             # dict
                "사이즈0": 가격0,       # int
                "사이즈1": 가격1        # int
            },
            "is_popular": 인기 메뉴 여부 # bool
        }
    ]
    """
    with open(path, "r") as f:
        return json.load(f)

def make_documents(items: List[Dict]) -> List[Document]:
    documents = []
    for item in items:
        document = Document(
            page_content=item["description"],
            metadata={key: value for key, value in item.items() if key != "description"}
        )
        documents.append(document)
    return documents

if __name__ == "__main__":
    import os
    
    # 경로 지정
    path = os.path.join(os.path.dirname(__file__), "fake_data.json")

    items = read_json(path) # JSON 파일 읽기
    documents = make_documents(items)  # Document 객체 생성

    # 테스트
    assert type(documents[0].page_content) == str, "설명은 str이여야 합니다."
    assert type(documents[0].metadata) == dict, "metadata는 dict여야 합니다."
    assert type(documents[0].metadata["pricing"]["L"]) == int, "가격은 int여야 합니다."
    assert type(documents[0].metadata["is_popular"]) == bool, "인기 메뉴 여부는 bool이여야 합니다."
    print("테스트 통과!")