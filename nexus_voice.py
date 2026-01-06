import streamlit as st
import os
from groq import Groq

client = None


def init_voice_client():
    global client
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        client = Groq(api_key=api_key)


def transcribe_audio(audio_bytes):
    if not client:
        init_voice_client()

    if not client or not audio_bytes:
        return None

    # Save temp file
    with open("temp_voice.wav", "wb") as f:
        f.write(audio_bytes)

    try:
        with open("temp_voice.wav", "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename("temp_voice.wav"), file.read()),
                model="whisper-large-v3-turbo",
                # 1. GUIDE THE AI: Tells it to expect a command, not a conversation
                prompt="User command for data analysis. Short technical query.",
                response_format="json",
                language="en",
                temperature=0.0
            )

        text = transcription.text.strip()

        # 2. FILTER HALLUCINATIONS: Block common whisper glitches
        hallucinations = [
            "Thank you.", "Thank you", "Thanks.", "You",
            "MBC", "Amara.org", "Subtitles by", "Copyright",
            "Thank you for watching"
        ]

        # If the output is just a hallucination, return empty so the app ignores it
        if text in hallucinations or len(text) < 2:
            return ""

        return text

    except Exception as e:
        st.error(f"Voice Error: {e}")
        return None