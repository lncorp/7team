from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# 1. 문서 로드
with open("db/gangwon_data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

docs = [Document(page_content=line.strip()) for line in lines if line.strip()]

# 임베딩 모델
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

# Chroma DB 생성
Chroma.from_documents(docs, embedding_model, persist_directory="db")