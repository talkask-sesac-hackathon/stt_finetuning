from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)
client = OpenAI()


def whisper_transcribe_from_file(audio_file_path: str | Path, prompt: str | None) -> str:
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            language="ko",
            model="whisper-1",
            file=audio_file,
            prompt=prompt if prompt else None
        )
    return transcript.text

def whisper_transcribe_from_audio(audio: bytes, prompt: str | None) -> str:
    transcript = client.audio.transcriptions.create(
        language="ko",
        model="whisper-1",
        file=audio,
        prompt=prompt if prompt else None
    )
    return transcript.text


if __name__ == "__main__":
    audio_file_path = "./음성메모.m4a"
    audio_file_path = Path("./음성메모.m4a")
    transcript = whisper_transcribe_from_file(audio_file_path, "구희찬")
    print(transcript)