from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# 1. 문서 로드
with open("db/gangwon_data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

docs = [Document(page_content=line.strip()) for line in lines if line.strip()]

# 2. 한국어 임베딩 모델 로드
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

# 3. 벡터 DB 생성 및 저장
db = FAISS.from_documents(docs, embedding_model)
db.save_local("faiss_gangwon_db")