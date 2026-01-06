import streamlit as st
import uuid
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# --- CUSTOM MODULES ---
from nexus_db import init_db, save_message, load_history, clear_session, get_all_sessions, save_setting, load_setting
from themes import THEMES, inject_theme_css
from nexus_engine import DataEngine
from nexus_brain import build_agent_graph, get_key_status

# --- UI CONFIG ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")
init_db()

# --- INITIALIZE STATE ---
if "data_engine" not in st.session_state: st.session_state.data_engine = DataEngine()
engine = st.session_state.data_engine

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
current_sess = st.session_state.current_session_id

# --- BUILD BRAIN ---
app = build_agent_graph(engine)

# --- SIDEBAR ---
current_theme = load_setting("theme", "🌿 Eywa (Avatar)")
inject_theme_css(current_theme)
theme_data = THEMES.get(current_theme, THEMES["🌿 Eywa (Avatar)"])

with st.sidebar:
    st.title("⚙️ NEXUS HQ")
    st.caption(get_key_status())

    uploaded_file = st.file_uploader("📂 Upload File", type=None)
    if uploaded_file:
        status = engine.load_file(uploaded_file)
        if "Error" in status:
            st.error(status)
        else:
            st.success(status)

    if st.button("🧹 Clear Plots"):
        plt.clf()
        engine.latest_figure = None
        st.success("Plots cleared.")

    st.divider()
    if st.button("➕ New Chat"):
        st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
        st.rerun()
    if st.button("🗑️ Clear Chat"):
        clear_session(current_sess)
        st.rerun()

    st.markdown("### 🕒 History")
    for s in get_all_sessions()[:5]:
        if st.button(f"{s}", key=s):
            st.session_state.current_session_id = s
            st.rerun()

# --- CHAT INTERFACE ---
st.title(f"NEXUS // {current_theme.split(' ')[1].upper()}")

# Load History
history = load_history(current_sess)
for msg in history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role, avatar=theme_data["user_avatar"] if role == "user" else theme_data["ai_avatar"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Enter command..."):
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        st.markdown(prompt)
    save_message(current_sess, "user", prompt)

    # Refresh Cheatsheet
    if engine.df is not None and not engine.column_str:
        engine.column_str = ", ".join(list(engine.df.columns))

    # --- SYSTEM PROMPT ---
    system_text = "You are Nexus, an advanced data analysis AI."
    has_data = "df" in engine.scope
    if has_data:
        system_text += f"""
        [DATA MODE ACTIVE]
        1. Variable 'df' is loaded.
        2. VALID COLUMNS: [{engine.column_str}]

        3. INSTRUCTIONS:
           - When asked for specific insights (Anomalies, Forecasts), prefer using the 'insights' module if applicable.
           - For general queries, write standard Python code using pandas/matplotlib.
           - ALWAYS explain the code output to the user clearly. Don't just show the number, interpret it.
           - If you generate a plot, mention "I have generated a chart above."
        """
    else:
        system_text += " If no file, use 'tavily' for web search."

    # Memory Window
    recent_history = history[-6:]

    messages = [SystemMessage(content=system_text)] + \
               [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in
                recent_history] + \
               [HumanMessage(content=prompt)]

    # Run Agent
    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Processing...", expanded=True)
        try:
            final_resp = ""
            # Loop through graph events
            for event in app.stream({"messages": messages}, config={"recursion_limit": 50}, stream_mode="values"):
                msg = event["messages"][-1]

                # Show Tool Calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for t in msg.tool_calls:
                        status_box.write(f"⚙️ Calling: `{t['name']}`")

                # Show Tool Output (Debug only, usually hidden in final UI)
                if isinstance(msg, ToolMessage):
                    status_box.write(f"⚙️ Data Received.")

                # Capture Final AI Response
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_resp = msg.content

            # 1. Render Chart if exists
            if engine.latest_figure:
                st.pyplot(engine.latest_figure)
                engine.latest_figure = None

            # 2. Render Final Text
            if final_resp:
                st.markdown(final_resp)
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_resp)
            else:
                # Fallback if no final text (rare)
                status_box.update(label="Complete", state="complete", expanded=False)

        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {e}")