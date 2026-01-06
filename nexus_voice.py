import streamlit as st
import os
import tempfile
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

    # ✅ FIX: Use /tmp/ directory for Streamlit Cloud permissions
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_voice.wav")

    # Save temp file
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    try:
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(temp_path), file.read()),
                model="whisper-large-v3-turbo",
                response_format="json",
                language="en",
                temperature=0.0
            )

        text = transcription.text.strip()

        # Minimalist Filter
        hallucinations = [
            "Thank you.", "Thank you", "Thanks.", "You", "MBC",
            "Amara.org", "Subtitles by", "Copyright", "Watching",
            "Thank you for watching"
        ]

        if text in hallucinations or text.lower().startswith("thank you for"):
            return ""

        return text

    except Exception as e:
        return None