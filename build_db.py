import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

excel_path = "db/ai_data.xlsx"
df = pd.read_excel(excel_path)

texts = []
for _, row in df.iterrows():
    age = row.get("2. 나이를 입력해주세요. (숫자만)", "")
    gender = row.get("1. 성별을 선택해주세요.", "")
    inconvenience = str(row.get("5. 강원도에서 관광/ 숙박 중 불편했던 점이 있었다면 자유롭게 기술해주세요.", "")).strip()
    help_case = str(row.get("6. 그 당시 도움을 받으셨던 사례가 있다면 자유롭게 작성해주세요. (예: 호텔직원, 관광안내소, 가이드 등)", "")).strip()
    expectation = str(row.get("7. AI 관광 챗봇을 사용할 때 가장 기대하는 기능을 선택해주세요.", "")).strip()
    extra = str(row.get("7-1. AI 관광 챗봇을 사용할 때 가장 기대하는 기능 기타의견 작성해주세요.", "")).strip()
    need = str(row.get("9. 강원도 관광 AI 챗봇이 제공했으면 하는 지역 관련 정보가 있다면 자유롭게 작성해주세요.", "")).strip()

    merged = f"응답자: {gender}, {age}세\n불편사항: {inconvenience}\n도움 사례: {help_case}\n기대 기능: {expectation}\n기타 의견: {extra}\n원하는 정보: {need}"
    texts.append(Document(page_content=merged))

embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
Chroma.from_documents(texts, embedding_model, persist_directory="db")
print("✅ 설문 기반 Chroma DB 구축 완료")