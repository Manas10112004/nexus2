import streamlit as st
import os
from groq import Groq

client = None

def init_voice_client():
    global client
    # Fetches key dynamically from environment (set by nexus_brain.py or secrets)
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        client = Groq(api_key=api_key)

def transcribe_audio(audio_file):
    """
    Transcribes audio directly from memory (BytesIO) without saving to disk.
    """
    if not client:
        init_voice_client()

    if not client or not audio_file:
        return None

    try:
        # 1. Reset file pointer to the beginning of the stream
        audio_file.seek(0)

        # 2. Send directly to API
        # The 'file' parameter accepts a tuple: (filename, file_object)
        # We use "input.wav" to hint the format to the API.
        transcription = client.audio.transcriptions.create(
            file=("input.wav", audio_file),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en",
            temperature=0.0
        )

        text = transcription.text.strip()

        # 3. Aggressive Glitch Filter
        # Whisper models sometimes hallucinate "Thank you" on empty audio.
        if not text or text.lower().strip() in ["thank you.", "thank you", "you"]:
            return ""

        return text

    except Exception as e:
        return f"[API Error: {str(e)}]"