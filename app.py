import streamlit as st
import openai
import os
from dotenv import load_dotenv
from streamlit_chat import message

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# 챗봇 응답 생성 함수

def chatbot_response(user_input, chat_history):
    messages = [{"role": "system", "content": "너는 카페에서 고객의 주문을 도와주는 챗봇이야. 최대한 친절하게 주문을 받아줘."}]
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

st.title("TalkASK 📢 ")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"user":'', "bot": "어서오세요. TalkAsk입니다 😊 무엇을 도와드릴까요? "}]

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

# 두 개의 열 생성
col1, col2 = st.columns([1, 3])

with col1:
    
    st.markdown("<h3 style='text-align: center;'>Talk Ask Chatbot🤖 </h3>", unsafe_allow_html=True)
    # 채팅 입력
    st.text_input("메시지를 입력하세요:", key="user_input", on_change=handle_input)

    # 채팅 내역 표시
    for i, entry in enumerate(st.session_state.chat_history):
        if entry['user']:  # 사용자 메시지가 비어있지 않은 경우에만 표시
            message(entry['user'], is_user=True, key=f"user_{i}")
        message(entry['bot'], key=f"bot_{i}")


with col2:
    st.title("메뉴판")
    # 필터링된 이미지 표시
    for image_path in st.session_state.filtered_images:
        st.image(image_path, use_column_width=True)

# 스타일 적용
st.markdown("""
<style>
    .chat-container {
        background-color: #DFF0D8; /* 채팅 컨테이너 배경색 */
        padding: 10px;
        border-radius: 10px;
    }
    .image-container {
        background-color: #FFFFFF; /* 이미지 컨테이너 배경색 */
        padding: 10px;
        border-radius: 10px;
    }
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
    .stTextInput>div>input {
        width: 100%;
    }
    .stButton>button {
        width: 100%;
    }
    .stColumn:nth-child(1) {
        background-color: #DFF0D8; /* 첫 번째 컬럼 배경색 */
        padding: 10px;
        border-radius: 10px;
    }
    .stColumn:nth-child(2) {
        background-color: #FFFFFF; /* 두 번째 컬럼 배경색 */
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)