import streamlit as st
import os
import operator
import torch
from typing import TypedDict, Annotated, Sequence
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIG ---
MODEL_SMART = "llama-3.3-70b-versatile"
DRAFT_MODEL_NAME = "meta-llama/Llama-3.2-1B" # Local Speed Demon

@st.cache_resource
def load_speed_demon():
    """Loads the local draft model onto the RTX 4060."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
        )
        return model, tokenizer
    except: return None, None

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def build_agent_graph(data_engine):
    draft_model, draft_tok = load_speed_demon()

    def agent_node(state):
        # UPGRADE 3: Speed Demon (Drafting)
        speculation = ""
        if draft_model:
            last_text = state["messages"][-1].content
            inputs = draft_tok(last_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                tokens = draft_model.generate(**inputs, max_new_tokens=10)
            speculation = draft_tok.decode(tokens[0], skip_special_tokens=True)
            # Tag the speculation for Groq to verify
            state["messages"].append(HumanMessage(content=f"[SPECULATIVE DRAFT]: {speculation}"))

        # VERIFICATION PHASE (Groq 70B)
        llm = ChatGroq(model=MODEL_SMART, temperature=0).bind_tools(
            [TavilySearchResults(max_results=2)], parallel_tool_calls=False
        )
        
        try:
            response = llm.invoke(state["messages"])
            if speculation: state["messages"].pop() # Clean state
            return {"messages": [response]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"Error: {e}")]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")
    return workflow.compile()
