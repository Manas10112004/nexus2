import streamlit as st
from supabase import create_client, Client
import os

# --- CONNECT TO CLOUD ---
# We try to get keys from Streamlit Secrets first, then Environment Variables
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

if not SUPABASE_URL or not SUPABASE_KEY:
    # Fallback to empty client to prevent crash on import, but will error on use
    print("⚠️ Supabase Credentials missing! Check secrets.")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def init_db():
    """Checks connection to Supabase."""
    if not supabase:
        st.error("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY.")
        st.stop()
    # No table creation needed here, we did it in the SQL Editor!


def save_message(session_id, role, content):
    """Saves a message to the cloud."""
    data = {
        "session_id": session_id,
        "role": role,
        "content": content
    }
    supabase.table("messages").insert(data).execute()


def load_history(session_id):
    """Loads chat history for a specific session from the cloud."""
    response = supabase.table("messages") \
        .select("*") \
        .eq("session_id", session_id) \
        .order("created_at", desc=False) \
        .execute()
    return response.data


def clear_session(session_id):
    """Deletes all messages for a specific session."""
    supabase.table("messages").delete().eq("session_id", session_id).execute()


def save_setting(key, value):
    """Upserts a setting (Update if exists, Insert if new)."""
    data = {"key": key, "value": value}
    supabase.table("settings").upsert(data).execute()


def load_setting(key, default_value):
    """Loads a setting, returns default if not found."""
    response = supabase.table("settings").select("value").eq("key", key).execute()

    if response.data:
        return response.data[0]['value']
    return default_value