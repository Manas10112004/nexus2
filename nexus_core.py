import streamlit as st
import uuid
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# Custom Modules
from nexus_db import init_db, save_message, load_history, clear_session, get_all_sessions, save_setting, load_setting
from themes import THEMES, inject_theme_css
from nexus_engine import DataEngine
from nexus_brain import build_agent_graph, get_key_status

# --- UI SETUP ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")
init_db()

if "data_engine" not in st.session_state: st.session_state.data_engine = DataEngine()
engine = st.session_state.data_engine

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
current_sess = st.session_state.current_session_id

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

# --- CHAT UI ---
st.title(f"NEXUS // {current_theme.split(' ')[1].upper()}")

history = load_history(current_sess)
for msg in history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role, avatar=theme_data["user_avatar"] if role == "user" else theme_data["ai_avatar"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter command..."):
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        st.markdown(prompt)
    save_message(current_sess, "user", prompt)

    # Refresh Columns
    if engine.df is not None and not engine.column_str:
        engine.column_str = ", ".join(list(engine.df.columns))

    system_text = "You are Nexus."
    has_data = "df" in engine.scope
    if has_data:
        system_text += f"""
        [DATA MODE]
        1. Variable 'df' is loaded.
        2. VALID COLUMNS: [{engine.column_str}]
        3. PLOTS: Use `plt.figure()` -> `plt.plot()`.
        4. STOPPING: If tool output says "[ANALYSIS COMPLETE]" or "[CHART GENERATED]", STOP.
        """
    else:
        system_text += " If no file, use 'tavily'."

    messages = [SystemMessage(content=system_text)] + \
               [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in
                history] + \
               [HumanMessage(content=prompt)]

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Processing...", expanded=True)
        try:
            final_resp = ""
            for event in app.stream({"messages": messages}, config={"recursion_limit": 50}, stream_mode="values"):
                msg = event["messages"][-1]
                if isinstance(msg, ToolMessage):
                    status_box.write(f"⚙️ Output: {msg.content[:200]}...")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for t in msg.tool_calls:
                        status_box.write(f"⚙️ Calling: `{t['name']}`")
                if isinstance(msg, AIMessage) and msg.content:
                    final_resp = msg.content
                    st.markdown(final_resp)

            # --- SHOW CHART (The Fix) ---
            if engine.latest_figure:
                st.pyplot(engine.latest_figure)
                engine.latest_figure = None  # Clear after showing

            if final_resp:
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_resp)
            else:
                st.error("No response.")

        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {e}")