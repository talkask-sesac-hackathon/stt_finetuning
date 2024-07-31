import streamlit as st
import openai
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# 챗봇 응답 생성 함수
def chatbot_response(user_input, chat_history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for entry in chat_history:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["bot"]})

    messages.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response['choices'][0]['message']['content'].strip()

# Streamlit 애플리케이션 설정
st.set_page_config(page_title="Chatbot Interface", layout="wide")

st.title("Image and Chatbot Interface")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "filtered_images" not in st.session_state:
    st.session_state.filtered_images = []

# 이미지 데이터 (예시)
image_data = {
    "category1": ["images/menu1.jpg", "images/menu2.jpg"],
    "category2": ["images/menu3.jpg", "images/menu4.jpg"],
    "category3": ["images/menu5.jpg", "images/menu6.jpg"]
}

# 사용자 입력 처리 콜백 함수
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"user": user_input, "bot": ""})
        
        # 챗봇 응답 생성
        bot_response = chatbot_response(user_input, st.session_state.chat_history)
        
        # 챗봇 응답 추가
        st.session_state.chat_history[-1]["bot"] = bot_response
        
        # 사용자의 입력을 기반으로 이미지 필터링 (예시로 간단한 키워드 필터링 사용)
        if "category1" in user_input:
            st.session_state.filtered_images = image_data["category1"]
        elif "category2" in user_input:
            st.session_state.filtered_images = image_data["category2"]
        elif "category3" in user_input:
            st.session_state.filtered_images = image_data["category3"]
        else:
            st.session_state.filtered_images = []

        # 입력 필드 초기화
        st.session_state.user_input = ""

# 사이드바에 채팅 인터페이스 추가
st.sidebar.title("Chatbot Interface")

# 채팅 입력
st.sidebar.text_input("메시지를 입력하세요:", key="user_input", on_change=handle_input)

# 채팅 내역 표시
for entry in st.session_state.chat_history:
    st.sidebar.markdown(f"<div class='user-message'>사용자: {entry['user']}</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='chatbot-message'>챗봇: {entry['bot']}</div>", unsafe_allow_html=True)

# 메인 영역에 필터링된 이미지 표시
st.title("Filtered Images")
for image_path in st.session_state.filtered_images:
    st.image(image_path, use_column_width=True)

# 스타일 적용
st.markdown("""
<style>
    .user-message {
        background-color: #dcf8c6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .chatbot-message {
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    .responsive-image img {
        width: 100%;
        height: auto;
    }
    /* 사이드바 너비 조정 */
    .css-1d391kg {
        width: 350px; /* 원하는 너비로 조정 */
    }
</style>
""", unsafe_allow_html=True)
