import streamlit as st
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Streamlit 설정
st.set_page_config(page_title="강원도 관광 및 숙박 특화 AI 챗봇", page_icon="🌄")
st.title("🌄 강원도 관광 및 숙박 특화 AI 챗봇")
st.markdown("질문 예시: `속초 명소 추천해줘`, `춘천에서 뭘 먹어야 해?`, `강릉 어디가 좋아?`")

# 사용자 질문 입력
question = st.text_input("✍️ 궁금한 점을 입력하세요:")

# QA 파이프라인 생성
qa_pipeline = pipeline(
    task='question-answering',
    model='beomi/KcELECTRA-base',
    tokenizer='beomi/KcELECTRA-base',
    device=-1  # CPU에서 강제로 실행
)

# Chroma DB 로딩
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
db = Chroma(persist_directory="db", embedding_function=embedding_model)

# 질문 처리
if question:
    with st.spinner("🔍 관련 문맥 검색 중..."):
        docs = db.similarity_search(question, k=1)

    if not docs:
        st.error("😥 관련 정보를 찾지 못했어요. 질문을 다시 입력해 주세요.")
    else:
        context = docs[0].page_content.strip()
        if not context:
            st.warning("문맥이 비어 있어 기본 설명으로 대체합니다.")
            context = "강원도는 자연, 바다, 산이 어우러진 대표 관광지입니다."

        with st.spinner("🤖 답변 생성 중..."):
            result = qa_pipeline(question=question, context=context, top_k=3)
            answers = [r["answer"] for r in result]
            st.markdown("### 🤖 챗봇의 답변")
            st.success(" · ".join(answers))