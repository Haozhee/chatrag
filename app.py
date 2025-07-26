import os, streamlit as st
from rag_chain import chat
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="ChatRAG", page_icon="🤖", layout="wide")

st.title("📚 ChatRAG – 私有知识库聊天")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# 聊天输入框
user_input = st.chat_input("提问或对话…")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("思考中…"):
        answer = chat(user_input)
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})