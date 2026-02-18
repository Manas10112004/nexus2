import streamlit as st
from supabase import create_client, Client

# --- CONNECTION MANAGER ---
@st.cache_resource
def get_supabase_client() -> Client:
    """Establishes a connection to Supabase using secrets."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Missing Database Secrets: {e}")
        st.stop()

def verify_user_plaintext(username, password):
    """
    Checks if a username and password match in the 'profiles' table.
    [DEV ONLY]: Uses plaintext comparison.
    """
    client = get_supabase_client()
    try:
        # Querying the profiles table for a direct match
        response = client.table("profiles")\
            .select("*")\
            .eq("username", username)\
            .eq("password", password)\
            .execute()
        
        if response.data and len(response.data) > 0:
            return True, response.data[0]
        return False, "❌ Invalid username or password."
    except Exception as e:
        return False, f"❌ Database Error: {str(e)}"
def init_db():
    """Verifies database connection on startup."""
    try:
        client = get_supabase_client()
        # Lightweight check to ensure table exists and we can connect
        client.table("chat_history").select("id", count="exact").limit(1).execute()
    except Exception as e:
        # Don't stop app, just log (tables might be empty)
        print(f"DB Check Warning: {e}")


# --- CHAT HISTORY FUNCTIONS ---

def save_message(session_id, role, content):
    """Saves a message to Supabase with the current username (email)."""
    client = get_supabase_client()
    # Fallback to 'guest' if not logged in, though Auth should prevent this
    username = st.session_state.get("user_email", "guest")

    data = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "username": username
    }
    client.table("chat_history").insert(data).execute()


def load_history(session_id):
    """Loads chat history for a specific session."""
    client = get_supabase_client()
    response = client.table("chat_history") \
        .select("*") \
        .eq("session_id", session_id) \
        .order("created_at", desc=False) \
        .execute()
    return response.data


def clear_session(session_id):
    """Deletes all messages for a specific session."""
    client = get_supabase_client()
    client.table("chat_history").delete().eq("session_id", session_id).execute()


def get_all_sessions():
    """Retrieves unique session IDs for the logged-in user."""
    client = get_supabase_client()
    username = st.session_state.get("user_email", "guest")

    # Only fetch sessions belonging to this user
    # Note: For large apps, use a dedicated 'sessions' table or .distinct() if available
    response = client.table("chat_history") \
        .select("session_id") \
        .eq("username", username) \
        .order("created_at", desc=True) \
        .execute()

    # Deduplicate session IDs manually (Supabase JS client supports .distinct, Python client varies)
    unique_sessions = []
    seen = set()
    for row in response.data:
        sid = row['session_id']
        if sid not in seen:
            unique_sessions.append(sid)
            seen.add(sid)

    return unique_sessions


# --- [UPDATED] USER PLAN MANAGEMENT ---

def get_user_plan(email):
    """
    Fetches the user's plan type from the 'profiles' table.
    Defaults to 'free' if the user is not found or error occurs.
    """
    if not email:
        return "free"

    client = get_supabase_client()
    try:
        # Query the 'profiles' table using the email column
        response = client.table("profiles").select("plan_type").eq("email", email).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0].get("plan_type", "free")
        return "free"
    except Exception as e:
        print(f"Plan Fetch Error: {e}")
        return "free"


def update_user_plan(email, new_plan):
    """Updates the user's plan (e.g., after payment)."""
    client = get_supabase_client()
    try:
        client.table("profiles").update({"plan_type": new_plan}).eq("email", email).execute()
        return True
    except Exception as e:
        print(f"Plan Update Error: {e}")
        return False

# ----------------------------------


# --- SETTINGS MANAGEMENT ---
# (Falling back to Session State for simplicity to avoid needing another SQL table)

def save_setting(key, value):
    st.session_state[f"setting_{key}"] = value


def load_setting(key, default):
    return st.session_state.get(f"setting_{key}", default)
