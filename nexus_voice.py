import streamlit as st
import os
from groq import Groq

client = None


def init_voice_client():
    global client
    # 1. Try Environment Variable
    api_key = os.environ.get("GROQ_API_KEY")

    # 2. Fallback to Secrets
    if not api_key:
        raw_keys = st.secrets.get("GROQ_API_KEYS", "")
        if raw_keys:
            api_key = raw_keys.split(",")[0].strip()

    if api_key:
        client = Groq(api_key=api_key)


def transcribe_audio(audio_file):
    if not client:
        init_voice_client()

    if not client:
        return "[System Error: API Key missing]"

    try:
        # Rewind file to start
        audio_file.seek(0)

        # Call API
        transcription = client.audio.transcriptions.create(
            file=("input.wav", audio_file),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en",
            temperature=0.0
        )

        text = transcription.text.strip()

        # --- FIX: STRICT FILTER REMOVED ---
        # We now return whatever the AI heard, even if it's "Thank you."
        # This helps you debug if the mic is too quiet.
        if not text:
            return ""

        return text

    except Exception as e:
        return f"[API Error: {str(e)}]"