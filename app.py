import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline

# LLM 설정
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)
llm = HuggingFacePipeline(pipeline=pipe)

# 벡터 DB 불러오기
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
db = FAISS.load_local("faiss_gangwon_db", embeddings)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Streamlit UI
st.title("🏔️ 강원도 관광 안내 챗봇 (LangChain + RAG)")
query = st.text_input("무엇이 궁금하신가요?")

if query:
    answer = qa.run(query)
    st.markdown(f"**🤖 챗봇:** {answer}")
