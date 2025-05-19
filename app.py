import streamlit as st

st.title("Hello 7팀2")
name = st.text_input("이름을 입력해주세요.")
if name != "":
    st.write(f"{name}님! 저희 팀에 오신걸 환영합니다.!!!!!!!!")   

#st.balloons()