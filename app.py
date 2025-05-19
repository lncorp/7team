import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline

# LLM 설정
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
#pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    do_sample=True,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=pipe)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    [아래는 강원도 관광 정보입니다. 이 문맥을 바탕으로 질문에 답하세요.]
    문맥:
    {context}

    질문:
    {question}

    답변:
    """
)


# 벡터 DB 불러오기
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
db = FAISS.load_local("db", embeddings, allow_dangerous_deserialization=True)
#qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# Streamlit UI
st.title("🏔️강원도 관광 및 숙박 특화 AI 챗봇")
query = st.text_input("무엇이 궁금하신가요?")

if query:
    answer = qa.run(query)
    st.markdown(f"**🤖 챗봇:** {answer}")
