import os
import shutil
import json
import time
import gc
import pandas as pd
from datetime import datetime

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

def upload_and_build_chroma(uploaded_file, base_db_dir="db", model_name="jhgan/ko-sbert-sts"):
    try:
        # 1. 파일 읽기
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.DataFrame(json.load(uploaded_file))
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            return False

        # 2. 문서 생성
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

        # 3. 유니크한 DB 디렉토리 생성 (예: db/db_session_20250603_153027)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_db_dir = os.path.join(base_db_dir, f"db_session_{timestamp}")
        os.makedirs(new_db_dir, exist_ok=True)

        # 4. Chroma 생성
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        chroma_db = Chroma.from_documents(texts, embedding_model, persist_directory=new_db_dir)

        # 5. 세션 상태에 저장
        st.session_state["chroma_db"] = chroma_db
        st.session_state["chroma_db_path"] = new_db_dir

        return True

    except Exception as e:
        st.error(f"❌ 업로드 처리 중 오류: {e}")
        return False
