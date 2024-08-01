'''
입력받은 텍스트를 음성 파일로 변환하고자 해요.
그리고, Streamlit 상에서 해당 파일을 재생하고자 해요.
'''
# !pip install pyttsx3

import pyttsx3

def do_TTS_and_play(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    # engine.save_to_file(text, 'output.wav')
    engine.say(text)
    engine.runAndWait()