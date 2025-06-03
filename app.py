import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from upload_file import upload_and_build_chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Streamlit 설정
st.set_page_config(page_title="강원도 관광 및 숙박 특화 AI 챗봇", page_icon="🌄")
st.title("🌄 강원도 관광 및 숙박 특화 AI 챗봇")
st.markdown("설문 기반으로 사람들의 의견을 반영한 응답을 제공합니다.")

# 🔁 GPT 모델 선택
model_display_names = {
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-4 Turbo (Mini)": "gpt-4o"
}
selected_model_display = st.selectbox("사용할 GPT 모델을 선택하세요", options=list(model_display_names.keys()))
selected_model = model_display_names[selected_model_display]

# 임베딩 모델 및 DB 불러오기
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
db = Chroma(persist_directory="db", embedding_function=embedding_model)

# 세션 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 사용자 질문 입력
question = st.chat_input("궁금한 점을 입력하세요:")

# 설문 파일 업로드
uploaded_file = st.file_uploader("설문 데이터 업로드 (xlsx/json)", type=["xlsx", "json"])
if uploaded_file:
    if upload_and_build_chroma(uploaded_file):
        st.success("업로드한 데이터로 새로운 Chroma DB가 생성되었습니다.")

# 질문 처리
if question and "chroma_db" in st.session_state:
    st.chat_message("user").write(question)
    with st.spinner("🤖 답변 생성 중입니다..."):
        docs = st.session_state["chroma_db"].similarity_search(question, k=3)

        if not docs:
            st.chat_message("assistant").write("❌ 관련 정보를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요.")
        else:
            context = docs[0].page_content
            try:
                messages = [
                    {"role": "system", "content": "당신은 강원도 관광에 대한 실제 설문 응답을 기반으로 설명하는 AI 챗봇입니다. 아래 문맥을 참고해 답변하세요."},
                    *[{"role": "user" if r == "질문" else "assistant", "content": m} for r, m in st.session_state.chat_history],
                    {"role": "user", "content": f"문맥: {context}\n질문: {question}"}
                ]

                response = client.chat.completions.create(
                    model=selected_model,  # 🔁 선택된 모델 사용
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7
                )
                answer = response.choices[0].message.content
                st.chat_message("assistant").write(answer)

                st.session_state.chat_history.append(("질문", question))
                st.session_state.chat_history.append(("답변", answer))
            except Exception as e:
                error_msg = str(e)
                if "insufficient_quota" in error_msg or "quota" in error_msg or "429" in error_msg:
                    st.warning("🚫 현재 API 사용량을 초과했습니다. OpenAI 설정을 확인해 주세요.")
                else:
                    st.error(f"❌ 오류 발생: {e}")

# 대화 기록 보기 및 저장
if st.session_state.chat_history:
    with st.expander("🗃️ 이전 질문과 답변 보기 / 저장 / 초기화"):
        for role, msg in st.session_state.chat_history:
            st.markdown(f"**{role.upper()}**: {msg}")
        st.download_button(
            label="📥 전체 대화 저장 (txt)",
            data="\n\n".join([f"{role.upper()}: {msg}" for role, msg in st.session_state.chat_history]),
            file_name="chat_history.txt",
            mime="text/plain"
        )
        if st.button("🧹 대화 기록 초기화"):
            st.session_state.chat_history.clear()
            st.success("✅ 기록이 초기화되었습니다.")
