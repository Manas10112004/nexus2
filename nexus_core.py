import streamlit as st
import pandas as pd
import os
import time
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

# Database Imports
from nexus_db import init_db, save_message, load_history, clear_session, save_setting, load_setting, get_all_sessions
from themes import THEMES, inject_theme_css

# --- 1. PAGE CONFIG (First Command) ---
st.set_page_config(page_title="Nexus AI", layout="wide", page_icon="⚡")

# --- 2. CONFIGURATION ---
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not TAVILY_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ System Halted: Missing API Keys in Streamlit Secrets.")
    st.stop()

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ✅ FIX 1: UPDATED MODEL NAME (Llama 3.1 is gone, 3.3 is the new standard)
MODEL_NAME = "llama-3.3-70b-versatile"


# --- 3. DATA ENGINE ---
class DataEngine:
    def __init__(self):
        self.df = None
        self.file_content = None
        self.file_type = None
        self.repl = PythonREPL()

    def load_file(self, uploaded_file):
        try:
            name = uploaded_file.name
            self.file_type = name.split('.')[-1].lower()
            if name.endswith(('.csv', '.xlsx', '.xls', '.json')):
                if name.endswith('.csv'):
                    self.df = pd.read_csv(uploaded_file)
                elif 'xls' in name:
                    self.df = pd.read_excel(uploaded_file)
                elif name.endswith('.json'):
                    self.df = pd.read_json(uploaded_file)
                return f"✅ Dataset loaded. Shape: {self.df.shape}. Available as variable 'df'."
            elif name.endswith(('.txt', '.py', '.md', '.log', '.toml', '.yml', '.yaml')):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                self.file_content = stringio.read()
                return f"✅ Text file read ({len(self.file_content)} chars)."
            else:
                return f"⚠️ File '{name}' received (Binary)."
        except Exception as e:
            return f"❌ Error loading file: {str(e)}"

    def run_python_analysis(self, code: str):
        try:
            local_scope = {"df": self.df, "file_content": self.file_content, "pd": pd, "plt": plt, "sns": sns, "st": st}
            return self.repl.run(code)
        except Exception as e:
            return f"Execution Error: {str(e)}"


# --- 4. SAFE INITIALIZATION ---
def get_data_engine():
    if "data_engine" not in st.session_state: st.session_state.data_engine = DataEngine()
    return st.session_state.data_engine


def get_session_id():
    if "current_session_id" not in st.session_state: st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
    return st.session_state.current_session_id


engine = get_data_engine()
current_sess = get_session_id()
init_db()

# --- 5. UI SETUP ---
current_theme_setting = load_setting("theme", "🌿 Eywa (Avatar)")
inject_theme_css(current_theme_setting)
theme_data = THEMES.get(current_theme_setting, THEMES["🌿 Eywa (Avatar)"])

with st.sidebar:
    st.title("⚙️ NEXUS HQ")
    uploaded_file = st.file_uploader("📂 Upload Data / Files", type=None)
    if uploaded_file:
        status = engine.load_file(uploaded_file)
        if "Error" in status:
            st.error(status)
        else:
            st.success(status)

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("➕ New Chat"):
        st.session_state.current_session_id = f"Session-{uuid.uuid4().hex[:4]}"
        st.rerun()
    if col2.button("🗑️ Clear"):
        clear_session(current_sess)
        st.rerun()

    st.markdown("### 🕒 History")
    all_sessions = get_all_sessions()
    for sess in all_sessions[:8]:
        if sess == current_sess:
            st.markdown(f"**🔹 {sess}**")
        else:
            if st.button(f"{sess}", key=f"btn_{sess}"):
                st.session_state.current_session_id = sess
                st.rerun()
    st.divider()
    web_search_on = st.toggle("🌐 Web Search", value=True)

# --- 6. AI GRAPH & TOOLS ---
tavily_tool = TavilySearchResults(max_results=2)
tools = [tavily_tool]


def python_analysis_tool(code: str):
    return engine.run_python_analysis(code)


# ENABLE TOOL IF DATA EXISTS
data_active = False
if engine.df is not None or engine.file_content is not None:
    data_active = True
    tools.append(Tool(
        name="python_analysis",
        func=python_analysis_tool,
        description="Run Python code on 'df' or 'file_content'."
    ))

llm = ChatGroq(model=MODEL_NAME, temperature=0.1)
llm_with_tools = llm.bind_tools(tools) if (web_search_on or len(tools) > 1) else llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def agent_node(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
if web_search_on or len(tools) > 1:
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
else:
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
app = workflow.compile()

# --- 7. CHAT INTERFACE ---
st.title(f"NEXUS // {current_theme_setting.split(' ')[1].upper()}")

history = load_history(current_sess)
current_messages = []

for msg in history:
    role = "user" if msg["role"] == "user" else "assistant"
    avatar = theme_data["user_avatar"] if role == "user" else theme_data["ai_avatar"]
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
    if role == "user":
        current_messages.append(HumanMessage(content=msg["content"]))
    else:
        current_messages.append(AIMessage(content=msg["content"]))

if prompt := st.chat_input("Enter command..."):
    with st.chat_message("user", avatar=theme_data["user_avatar"]):
        st.markdown(prompt)
    save_message(current_sess, "user", prompt)

    # ✅ FIX 2: FORCE DATA CONTEXT INTO THE SYSTEM PROMPT
    data_context = ""
    if engine.df is not None:
        data_context = f"""
        [DATA DETECTED]
        A dataframe named 'df' is loaded in memory.
        - Shape: {engine.df.shape}
        - Columns: {list(engine.df.columns)}

        INSTRUCTIONS:
        You MUST use the 'python_analysis' tool to answer questions about this data.
        Example: If user asks "Total revenue", write python code: df['Revenue'].sum()
        """
    elif engine.file_content is not None:
        data_context = f"[FILE DETECTED] A text file is loaded in variable 'file_content'. Use python_analysis to print it."

    system_text = f"""You are Nexus, an elite Data Scientist.
    {data_context}

    If no data is relevant, use 'tavily' to search the web.
    """

    # We insert the system message right before the user's latest query
    current_messages.append(SystemMessage(content=system_text))
    current_messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar=theme_data["ai_avatar"]):
        status_box = st.status("Processing...", expanded=True)
        message_placeholder = st.empty()
        final_response = ""

        try:
            for event in app.stream({"messages": current_messages}, stream_mode="values"):
                last_msg = event["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and len(last_msg.tool_calls) > 0:
                    for t in last_msg.tool_calls:
                        status_box.write(f"⚙️ **Using Tool:** `{t['name']}`")

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
                    message_placeholder.markdown(final_response)

            if final_response:
                status_box.update(label="Complete", state="complete", expanded=False)
                save_message(current_sess, "assistant", final_response)
            else:
                status_box.update(label="Failed", state="error")
                st.error("No response.")

        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(f"Error: {str(e)}")