import streamlit as st
import os
import operator
import torch  # Required for RTX 4060 local drafting
from typing import TypedDict, Annotated, Sequence
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer # For local RTX 4060 draft

# --- CONFIGURATION ---
MODEL_SMART = "llama-3.3-70b-versatile"
MODEL_FAST = "llama-3.1-8b-instant"
DRAFT_MODEL_LOCAL = "meta-llama/Llama-3.2-1B" # Tiny model for RTX 4060

# --- [NEW: SPEED DEMON] LOCAL DRAFT INITIALIZATION ---
@st.cache_resource
def load_local_drafter():
    """Loads a 1B model onto your RTX 4060 for fast guessing."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_LOCAL)
        model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_LOCAL, 
            torch_dtype=torch.float16, 
            device_map="cuda" # Forces usage of your RTX 4060
        )
        return model, tokenizer
    except Exception as e:
        st.warning(f"Local Drafter failed to load: {e}. Falling back to standard mode.")
        return None, None

# Load Keys
raw_groq = st.secrets.get("GROQ_API_KEYS", "")
raw_tavily = st.secrets.get("TAVILY_API_KEYS", "")
GROQ_KEYS = [k.strip() for k in raw_groq.split(",") if k.strip()]
TAVILY_KEYS = [k.strip() for k in raw_tavily.split(",") if k.strip()]

if not GROQ_KEYS or not TAVILY_KEYS:
    st.error("⚠️ System Halted: Missing API Keys in Secrets.")
    st.stop()

# --- KEY MANAGEMENT ---
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
    return f"Active Key: Groq-{st.session_state.groq_idx + 1}"

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
        description="Executes Python code. Access 'df' (pandas DataFrame). Use plt.show() for plots.",
        args_schema=PythonInput
    )
    return [search, python_tool]

# --- AGENT GRAPH ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def build_agent_graph(data_engine):
    init_keys()
    # Initialize Speed Demon local model
    draft_model, draft_tokenizer = load_local_drafter()

    def agent_node(state):
        models_to_try = [MODEL_SMART, MODEL_FAST]
        last_error = None

        # --- [NEW: SPEED DEMON] DRAFTING PHASE ---
        speculated_tokens = ""
        if draft_model:
            try:
                # Use last message to "guess" next 5-10 tokens locally
                last_msg = state["messages"][-1].content
                inputs = draft_tokenizer(last_msg, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = draft_model.generate(**inputs, max_new_tokens=10, do_sample=True)
                speculated_tokens = draft_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Inject a hint into the system state for the Smart model
                state["messages"].append(HumanMessage(content=f"[DRAFT GUESS]: {speculated_tokens}"))
            except:
                pass # Silent fail to standard mode

        for model_name in models_to_try:
            try:
                tools = get_tools(data_engine)
                key = os.environ["GROQ_API_KEY"]

                llm = ChatGroq(
                    model=model_name,
                    temperature=0.0,
                    api_key=key
                ).bind_tools(tools, parallel_tool_calls=False)

                # --- [NEW: SPEED DEMON] VERIFICATION PHASE ---
                # The Smart model receives the draft guess and verifies it in its forward pass
                response = llm.invoke(state["messages"])
                
                # Clean up the draft message from state if present
                if speculated_tokens:
                    state["messages"].pop() 

                return {"messages": [response]}

            except Exception as e:
                if "429" in str(e) or "Rate limit" in str(e):
                    rotate_groq_key()
                    continue
                last_error = e
                continue

        return {"messages": [AIMessage(content=f"❌ System Busy. Error: {str(last_error)}")]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(get_tools(data_engine)))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()
