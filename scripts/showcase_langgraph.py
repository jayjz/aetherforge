"""
AetherForge LangGraph Showcase
==============================
Demonstrates an autonomous agent checking physical VRAM pressure and 
mathematically justifying a hardware strategy swap before executing a task.
"""

import os
import requests
from typing import Annotated, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# AetherForge Control Plane URL
AETHERFORGE_URL = "http://127.0.0.1:8000"

# --- 1. DEFINE THE AETHERFORGE HARDWARE TOOLS ---

@tool
def check_hypervisor_metrics() -> str:
    """Always call this first. Checks live VRAM pressure, active strategy, and hardware generation speeds (TPS)."""
    try:
        response = requests.get(f"{AETHERFORGE_URL}/system/metrics")
        data = response.json()
        pressure = data["vram_pressure"]["utilization_pct"]
        mode = data["active_strategy"]
        return f"Hypervisor State: {mode.upper()} mode active. VRAM Pressure: {pressure:.1f}%."
    except Exception as e:
        return f"Failed to reach hypervisor: {e}"

@tool
def optimize_vram_strategy(mode: str, expected_output_tokens: int) -> str:
    """
    Call this to reallocate hardware VRAM before heavy tasks.
    Modes: 
    - 'high_fidelity': For dense coding, logic, and deep reasoning.
    - 'balanced': For standard chat and routing.
    - 'aggressive_quant': For summarization to clear VRAM.
    """
    payload = {
        "mode": mode,
        "estimated_context_tokens": 500, # Hardcoded for showcase simplicity
        "expected_output_tokens": expected_output_tokens
    }
    try:
        response = requests.post(f"{AETHERFORGE_URL}/system/strategy", json=payload)
        result = response.json()
        if result.get("status") == "rejected":
            return f"Gatekeeper rejected swap: {result.get('reason')}. Remaining in {result.get('active_mode')}."
        return f"Hardware Fast-Swap successful. Now operating in {result.get('active_mode').upper()} mode."
    except Exception as e:
        return f"Hardware exception: {e}"

tools = [check_hypervisor_metrics, optimize_vram_strategy]

# --- 2. BUILD THE LANGGRAPH AGENT ---

# NOTE: You need an OPENAI_API_KEY exported in your terminal for this brain to work, 
# or point it to a local vLLM/Ollama proxy via base_url.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState):
    """The brain of the operation."""
    system_prompt = SystemMessage(content=(
        "You are an autonomous AI engineer with physical control over your own GPU VRAM via AetherForge. "
        "Before writing any code or doing complex logic, you MUST: "
        "1. Check the hypervisor metrics. "
        "2. Optimize your VRAM strategy to 'high_fidelity'. "
        "Once the hardware is primed, answer the user's prompt."
    ))
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal["tools", END]:
    """Router to determine if the agent wants to pull a hardware lever."""
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- 3. COMPILE THE ORCHESTRATOR ---

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- 4. EXECUTE THE SHOWCASE ---

if __name__ == "__main__":
    print("🤖 Booting LangGraph Agent orchestrator...")
    print("Connecting to AetherForge Hypervisor...\n")
    
    user_task = "We need to build a complex quantitative finance scraper in Python using Playwright. Prepare your hardware and give me the architectural plan."
    
    print(f"USER REQUEST: '{user_task}'\n")
    print("-" * 50)
    
    inputs = {"messages": [HumanMessage(content=user_task)]}
    
    for event in app.stream(inputs, stream_mode="values"):
        message = event["messages"][-1]
        
        # Print Tool Calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                print(f"[Agent Intention] 🛠️ Calling Hypervisor Tool: {tc['name']} with args {tc['args']}")
        
        # Print Tool Results
        elif message.type == "tool":
            print(f"[Hypervisor Execution] ⚙️ Result: {message.content}\n")
            
    print("-" * 50)
    print(f"\n[Final Agent Output]\n{event['messages'][-1].content}")