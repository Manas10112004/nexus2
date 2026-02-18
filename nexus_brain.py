import streamlit as st
import os
import operator
import torch
from typing import TypedDict, Annotated, Sequence
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_SMART = "llama-3.3-70b-versatile"
DRAFT_MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Load Keys
raw_groq = st.secrets.get("GROQ_API_KEYS", "")
raw_tavily = st.secrets.get("TAVILY_API_KEYS", "")
GROQ_KEYS = [k.strip() for k in raw_groq.split(",") if k.strip()]
TAVILY_KEYS = [k.strip() for k in raw_tavily.split(",") if k.strip()]

if not GROQ_KEYS or not TAVILY_KEYS:
    st.error("⚠️ System Halted: Missing API Keys in Secrets.")
    st.stop()

# --- KEY MANAGEMENT (Restored) ---
def init_keys():
    if "groq_idx" not in st.session_state: st.session_state.groq_idx = 0
    if "tavily_idx" not in st.session_state: st.session_state.tavily_idx = 0
    update_env_vars()

def update_env_vars():
    g_key = GROQ_KEYS[st.session_state.groq_idx % len(GROQ_KEYS)]
    t_key = TAVILY_KEYS[st.session_state.tavily_idx % len(TAVILY_KEYS)]
    os.environ["GROQ_API_KEY"] = g_key
    os.environ["TAVILY_API_KEY"] = t_key

def rotate_groq_key():
    st.session_state.groq_idx = (st.session_state.groq_idx + 1) % len(GROQ_KEYS)
    update_env_vars()

def get_key_status():
    """Returns the status string for nexus_core.py"""
    return f"Active Key: Groq-{st.session_state.groq_idx + 1}"

# --- SPEED DEMON LOADER ---
@st.cache_resource
def load_speed_demon():
    try:
        tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
        )
        return model, tokenizer
    except: return None, None

# --- AGENT SETUP ---
class PythonInput(BaseModel):
    code: str = Field(description="Python code to execute. Always print output.")

def get_tools(data_engine):
    update_env_vars()
    search = TavilySearchResults(max_results=2)
    def python_wrapper(code: str):
        return data_engine.run_python_analysis(code)
    python_tool = StructuredTool.from_function(
        func=python_wrapper,
        name="python_analysis",
        description="Executes Python code. Access 'df' (pandas DataFrame).",
        args_schema=PythonInput
    )
    return [search, python_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- BUILD BRAIN ---
def build_agent_graph(data_engine):
    init_keys()
    draft_model, draft_tok = load_speed_demon()

    def agent_node(state):
        # SPEED DEMON: Drafting
        speculation = ""
        if draft_model:
            try:
                last_text = state["messages"][-1].content
                inputs = draft_tok(last_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    tokens = draft_model.generate(**inputs, max_new_tokens=10)
                speculation = draft_tok.decode(tokens[0], skip_special_tokens=True)
                state["messages"].append(HumanMessage(content=f"[SPECULATIVE DRAFT]: {speculation}"))
            except: pass

        # VERIFICATION
        try:
            tools = get_tools(data_engine)
            llm = ChatGroq(model=MODEL_SMART, temperature=0).bind_tools(tools, parallel_tool_calls=False)
            response = llm.invoke(state["messages"])
            if speculation: state["messages"].pop() 
            return {"messages": [response]}
        except Exception as e:
            if "429" in str(e):
                rotate_groq_key()
                return agent_node(state) # Retry with new key
            return {"messages": [AIMessage(content=f"Error: {e}")]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(get_tools(data_engine)))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()
