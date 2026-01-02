import streamlit as st
import os
import time
import uuid
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

# Imports from your database file
# Make sure get_all_sessions is added to nexus_db.py!
from nexus_db import init_db, save_message, load_history, clear_session, save_setting, load_setting, get_all_sessions
from themes import THEMES, inject_theme_css

# --- CONFIGURATION & SECRETS ---
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not TAVILY_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ System Halted: Missing API Keys in secrets.")
    st.stop()

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

MODEL_NAME = "llama-3.3-70b-versatile"


# --- PDF GENERATOR ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'NEXUS RESEARCH REPORT', 0, 1, 'C')
        self.ln(5)
        self.line(10, 25, 200, 25)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        clean_body = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 6, clean_body)
        self.ln()


def create_pdf_report(content, filename="report.pdf"):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_body(content)
    pdf_path = f"/tmp/{filename}"
    pdf.output(pdf_path)
    return pdf_path


# --- MAIN APP UI & SETUP ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")
init_db()

# --- SESSION MANAGEMENT LOGIC ---
if "current_session_id" not in st.session_state:
    # Generates a unique ID like "Session-a1b2"
    st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"

# Theme Fallback
current_theme_setting = load_setting("theme", "🌿 Eywa (Avatar)")
if current_theme_setting not in THEMES:
    current_theme_setting = "🌿 Eywa (Avatar)"

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ SYSTEM")

    # 1. NEW CHAT BUTTON
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
        st.rerun()

    st.divider()

    # 2. SESSION HISTORY LIST
    st.markdown("### 🕒 History")
    all_sessions = get_all_sessions()

    # Show last 10 sessions to keep sidebar clean
    for sess in all_sessions[:10]:
        # Highlight the active session
        if sess == st.session_state.current_session_id:
            st.markdown(f"**🔹 {sess}**")
        else:
            if st.button(f"{sess}", key=f"btn_{sess}"):
                st.session_state.current_session_id = sess
                st.rerun()

    st.divider()

    # 3. SETTINGS
    selected_theme = st.selectbox("Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(current_theme_setting))
    if selected_theme != current_theme_setting:
        save_setting("theme", selected_theme)
        st.rerun()

    # --- SEARCH LOGIC ---
    if "deep_research" not in st.session_state:
        st.session_state.deep_research = False


    def on_deep_research_change():
        if st.session_state.deep_research:
            st.session_state.web_search = True


    deep_research_mode = st.toggle("Deep Research", key="deep_research", on_change=on_deep_research_change)
    is_search_forced = deep_research_mode

    web_search_on = st.toggle(
        "🌐 Web Search",
        key="web_search",
        value=True if is_search_forced else None,
        disabled=is_search_forced
    )

    enable_search = web_search_on or deep_research_mode

    st.divider()
    if st.button("🗑️ Delete Current Chat"):
        clear_session(st.session_state.current_session_id)
        st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
        st.rerun()

inject_theme_css(selected_theme)
theme_data = THEMES[selected_theme]

# Display Current Session Name
st.caption(f"Current ID: {st.session_state.current_session_id}")
st.title(f"NEXUS // {selected_theme.split(' ')[1].upper()}")

# --- DYNAMIC AI GRAPH SETUP ---
tool = TavilySearchResults(max_results=2)
tools = [tool]

llm = ChatGroq(model=MODEL_NAME, temperature=0.2)

if enable_search:
    llm_with_tools = llm.bind_tools(tools)
else:
    llm_with_tools = llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent_node(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)

if enable_search:
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
else:
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

app = workflow.compile()

# --- CHAT INTERFACE ---
# Load history for the *current* session ID
history = load_history(st.session_state.current_session_id)
current_messages = []

for msg in history:
    role = "user" if msg["role"] == "user" else "assistant"
    avatar = theme_data["user_avatar"] if role == "user" else theme_data["ai_avatar"]

    if role == "user":
        current_messages.append(HumanMessage(content=msg["content"]))
    else:
        current_messages.append(AIMessage(content=msg["content"]))

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter command..."):
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        st.markdown(prompt)
    save_message(st.session_state.current_session_id, "user", prompt)

    if deep_research_mode:
        research_prompt = """
        You are a Senior Python Software Engineer.
        User Query: {prompt}
        Provide robust, production-ready code using modern libraries.
        """
        current_messages.append(SystemMessage(content=research_prompt.format(prompt=prompt)))
    else:
        current_messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Processing...", expanded=True)
        message_placeholder = st.empty()
        final_response = ""

        try:
            for event in app.stream({"messages": current_messages}, stream_mode="values"):
                last_msg = event["messages"][-1]

                if enable_search:
                    if hasattr(last_msg, 'tool_calls') and len(last_msg.tool_calls) > 0:
                        for t in last_msg.tool_calls:
                            status_box.write(f"🔍 **Searching:** {t['args']}")
                    elif isinstance(last_msg, ToolMessage):
                        status_box.write("✅ **Results Found.** Analyzing...")

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
                    message_placeholder.markdown(final_response)

            if not final_response and enable_search:
                status_box.write("⚠️ **Synthesizing Data...**")
                nudge_message = HumanMessage(content="Summarize the search results now.")
                forced_response = llm.invoke(current_messages + [nudge_message])
                final_response = forced_response.content
                message_placeholder.markdown(final_response)

            if final_response:
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(st.session_state.current_session_id, "assistant", final_response)

                if deep_research_mode:
                    pdf_path = create_pdf_report(final_response, f"report_{int(time.time())}.pdf")
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button("📄 Download Report", pdf_file, "Nexus_Report.pdf", "application/pdf")
            else:
                status_box.update(label="Failed", state="error")
                st.error("Model failed to generate text.")

        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {str(e)}")