import streamlit as st
import os
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="강원도 관광 및 숙박 특화 AI 챗봇", page_icon="🌄")
st.title("🌄 강원도 관광 및 숙박 특화 AI 챗봇")
st.markdown("설문 기반으로 사람들의 의견을 반영한 응답을 제공합니다.")

embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
db = Chroma(persist_directory="db", embedding_function=embedding_model)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("✍️ 궁금한 점을 입력하세요:")

if question:
    st.chat_message("user").write(question)
    with st.spinner("🔍 관련 문맥 검색 중..."):
        docs = db.similarity_search(question, k=1)
    context = docs[0].page_content if docs else "❌ 관련 정보를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요."

    try:
        messages = [
            {"role": "system", "content": "당신은 강원도 관광에 대한 실제 설문 응답을 기반으로 설명하는 AI 챗봇입니다. 아래 문맥을 참고해 답변하세요."},
            *st.session_state.chat_history,
            {"role": "user", "content": f"문맥: {context}\n질문: {question}"}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        answer = response.choices[0].message.content
        st.chat_message("assistant").write(answer)

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "quota" in error_msg or "429" in error_msg:
            st.warning("🚫 현재 API 사용량을 초과했습니다. OpenAI 설정을 확인해 주세요.")
        else:
            st.error(f"❌ 오류 발생: {e}")

if st.session_state.chat_history:
    with st.expander("💬 대화 기록"):
        for msg in st.session_state.chat_history:
            role = "사용자" if msg["role"] == "user" else "GPT"
            st.markdown(f"**{role}:** {msg['content']}")
        if st.button("🧹 기록 초기화"):
            st.session_state.chat_history.clear()
