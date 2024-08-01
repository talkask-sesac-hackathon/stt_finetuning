from pathlib import Path
from typing import List, Dict
import json

from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_core.embeddings.embeddings import Embeddings


def read_json(path: str | Path) -> List[Dict[str, str | bool | dict[str, int]]]:
    """ JSON 데이터 구조
    [
        {
            "name": "메뉴 이름",           # str
            "description": "설명",        # str
            "category": "음료 종류",       # str
            "is_popular": 인기 메뉴 여부,   # bool
            "temperature": "온도",        # str
            "has_caffeine": 카페인 함유 여부,# bool
            "pricing": {                 # dict
                "사이즈0": 가격0,           # int
                "사이즈1": 가격1            # int
            }
        }
    ]
    """
    with open(path, "r") as f:
        return json.load(f)

def make_documents(items: List[Dict]) -> List[Document]:
    documents = []
    for item in items:
        # metadata에서 "description"을 제외한 모든 항목을 가져오되,
        # "pricing"이 dict 형태인 경우 이를 JSON 문자열로 변환
        metadata = {}
        for key, value in item.items():
            if key != "description":
                if key == "pricing" and isinstance(value, dict):
                    # dict를 JSON 문자열로 변환
                    metadata[key] = json.dumps(value, ensure_ascii=False)
                else:
                    metadata[key] = value
        
        # Document 객체 생성
        document = Document(
            page_content=item["description"],
            metadata=metadata
        )
        documents.append(document)
    return documents

def create_chroma(
        documents: List[Document],
        embedding: Embeddings,
        persist_directory: str | Path
        ) -> Chroma:
    """
    주어진 Document 리스트와 임베딩을 사용하여 Chroma DB를 생성하고 저장합니다.

    Args:
        documents (list): LangChain Document 객체 리스트.
        embedding: 임베딩 객체 (OpenAIEmbeddings).
        persist_path (str): Chroma DB를 저장할 디렉토리 경로.

    Returns:
        chroma: 생성된 Chroma 벡터스토어 객체.
    """
    embedding = OpenAIEmbeddings() if embedding is None else embedding
    chroma = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
    chroma.add_documents(documents)
    return chroma

def read_chroma(persist_directory: str | Path, embedding: Embeddings) -> Chroma:
    """
    주어진 디렉토리 경로에서 Chroma DB를 읽어옵니다.

    Args:
        persist_path (str): Chroma DB가 저장된 디렉토리 경로.
        embedding: 임베딩 객체 (OpenAIEmbeddings).
    Returns:
        chroma: 읽어온 Chroma 벡터스토어 객체.
    """
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)

def parse_document(document: Document) -> dict:
    metadata = document.metadata
    if "pricing" in metadata:
        metadata["pricing"] = json.loads(metadata["pricing"])
    metadata["description"] = document.page_content
    return metadata

def parse_documents(documents: List[Document]) -> List[dict]:
    return [parse_document(document) for document in documents]


if __name__ == "__main__":
    import os
    
    # 경로 지정
    path = os.path.join(os.path.dirname(__file__), "fake_data.json")

    items = read_json(path) # JSON 파일 읽기
    documents = make_documents(items)  # Document 객체 생성

    # 테스트
    assert type(documents[0].page_content) == str, "설명은 str이여야 합니다."
    assert type(documents[0].metadata) == dict, "metadata는 dict여야 합니다."
    assert type(json.loads(documents[0].metadata["pricing"])["L"]) == int, "가격은 int여야 합니다."
    assert type(documents[0].metadata["is_popular"]) == bool, "인기 메뉴 여부는 bool이여야 합니다."
    print("데이터 형식 테스트 통과!")

    # Embedding 설정
    from dotenv import load_dotenv    
    load_dotenv(override=True)
    embedding = OpenAIEmbeddings()
    chroma_path = os.path.join(os.path.dirname(__file__), "chroma")
    
    # Chroma DB 생성
    # chroma = create_chroma(documents, embedding=embedding, persist_directory=chroma_path)
    # print("Chroma DB 생성 완료!")
    print()

    # Chroma DB 읽기
    chroma = read_chroma(chroma_path, embedding)
    print("Chroma DB 읽기 완료!")
    
    # 검색
    query = "시원한 커피"
    results = chroma.similarity_search(query, k=4, filter={"is_popular": True})
    
    from pprint import pprint
    pprint(parse_documents(results))
    print("검색 완료!")
    