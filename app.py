import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# 페이지 기본 설정
st.set_page_config(page_title="강원도 관광 및 숙박 특화 AI 챗봇", page_icon="🌄")
st.title("🌄 강원도 관광 및 숙박 특화 AI 챗봇")
st.markdown("질문 예시: `속초 명소 추천해줘`, `춘천에서 뭘 먹어야 해?`, `강릉 어디가 좋아?`")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# KcELECTRA(케이씨 일렉트라) 모델 로딩
#@st.cache_resource
#def load_qa_pipeline():
#    return pipeline(
#        task="question-answering",
#        model="beomi/KcELECTRA-base",
#        tokenizer="beomi/KcELECTRA-base",
#        device=-1  # CPU 전용 실행
#    )

#qa = load_qa_pipeline()

# KoAlpaca(코알파카) 모델 로딩
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
    model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

qa = load_model()

# Chroma 벡터 DB 로딩
@st.cache_resource
def load_chroma():
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
    return Chroma(persist_directory="db", embedding_function=embedding_model)

db = load_chroma()

# 사용자 질문 입력
question = st.chat_input("✍️ 궁금한 점을 입력하세요:")


# 질문 처리
if question:
    with st.spinner("🔍 관련 문맥 검색 중..."):
        docs = db.similarity_search(question, k=1)

    if not docs:
        st.chat_message("assistant").write("❌ 관련 정보를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요.")
    else:
        context = docs[0].page_content.strip()
        with st.spinner("🤖 답변 생성 중입니다..."):
            try:
                prompt = f"### 질문: {question}\n### 문맥: {context}\n### 답변:"
                output = qa(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
                answer = output[0]["generated_text"].split("### 답변:")[-1].strip()

                st.chat_message("user").write(question)
                st.chat_message("assistant").write(answer)

                # 히스토리 저장
                st.session_state.chat_history.append(("질문", question))
                st.session_state.chat_history.append(("답변", answer))

            except Exception as e:
                st.chat_message("assistant").write(f"⚠️ 오류 발생: {e}")

# 이전 대화 기록 보기 / 다운로드 / 초기화
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