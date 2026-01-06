import streamlit as st
import os
from groq import Groq

client = None


def init_voice_client():
    global client
    # 1. Try Environment Variable (set by nexus_brain)
    api_key = os.environ.get("GROQ_API_KEY")

    # 2. Fallback: Try Secrets directly (Safety Net)
    if not api_key:
        raw_keys = st.secrets.get("GROQ_API_KEYS", "")
        if raw_keys:
            api_key = raw_keys.split(",")[0].strip()

    if api_key:
        try:
            client = Groq(api_key=api_key)
        except Exception as e:
            st.error(f"Groq Client Init Failed: {e}")
            client = None
    else:
        # Debug Warning if Key is missing
        print("DEBUG: No GROQ_API_KEY found in Env or Secrets.")


def transcribe_audio(audio_file):
    if not client:
        init_voice_client()

    if not client:
        return "[Error: API Key missing. Check st.secrets]"

    if not audio_file:
        return "[Error: Audio file is empty]"

    try:
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

        # DEBUG: Return raw text even if it looks like a glitch
        if not text:
            return "[Error: API returned empty text]"

        return text

    except Exception as e:
        return f"[API Error: {str(e)}]"