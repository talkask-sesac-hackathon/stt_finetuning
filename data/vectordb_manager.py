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


if __name__ == "__main__":
    path = Path("./fake_data.json")
    data = read_json(path)
