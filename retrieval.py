import os
import json
from typing import Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain.schema.document import Document

from data.vectordb_manager import read_chroma

def document_retrieval(
        query: str,
        vector_store: Chroma | VectorStore,
        k: int = 1
    ) -> Document:
    document = vector_store.similarity_search(query, k=k)[0]
    return document

def document_retrieval_with_score(
        query: str,
        vector_store: Chroma | VectorStore,
        k: int = 1
    ) -> Tuple[Document, float]:
    document, score = vector_store.similarity_search_with_score(query, k=k)[0]
    return document, score

def parse_document(document: Document) -> dict:
    metadata = document.metadata
    if "pricing" in metadata:
        metadata["pricing"] = json.loads(metadata["pricing"])
    metadata["description"] = document.page_content
    return metadata

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        load_dotenv(override=True)
    
    chroma_path = os.path.join(os.path.dirname(__file__), "data", "chroma")
    embedding = OpenAIEmbeddings()
    
    chroma = read_chroma(chroma_path, embedding)
    document, score = document_retrieval_with_score("커피", chroma)
    # document = document_retrieval("커피", chroma)
    metadata = parse_document(document)
    print(f"{metadata = }")
    print(f"{score = }")