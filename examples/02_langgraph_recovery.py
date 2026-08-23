"""
AetherForge LangGraph Showcase (Phase C)
========================================
Demonstrates an autonomous agent interacting with the AetherForge SDK.
It intentionally triggers a 413 Context Exceeded error, catches the rejection,
and routes to an autonomous truncation fallback node to recover without crashing.
"""

import os
import json
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# --- AUTO-GENERATED SDK IMPORTS ---
from aether_forge_hypervisor_api_client import Client
from aether_forge_hypervisor_api_client.models import StrategyPayload, GenerationPayload
from aether_forge_hypervisor_api_client.api.default import (
    get_metrics_system_metrics_get,
    update_strategy_system_strategy_post,
    generate_text_generate_post
)
    get_metrics_system_metrics_get,
    update_strategy_system_strategy_post,
    generate_text_generate_post
)

# Initialize the auto-generated SDK client
aether_client = Client(base_url="http://127.0.0.1:8000")

# --- 1. DEFINE THE AETHERFORGE HARDWARE TOOLS ---

@tool
def check_hypervisor_metrics() -> str:
    """Always call this first. Checks live VRAM pressure, active strategy, and generation speeds (TPS)."""
    try:
        res = get_metrics_system_metrics_get.sync_detailed(client=aether_client)
        if res.status_code == 200:
            data = json.loads(res.content)
            pressure = data["vram_pressure"]["utilization_pct"]
            mode = data["active_strategy"]
            return f"Hypervisor State: {mode.upper()} mode active. VRAM Pressure: {pressure:.1f}%."
        return f"Failed: HTTP {res.status_code}"
    except Exception as e:
        return f"Failed to reach hypervisor: {e}"

@tool
def optimize_vram_strategy(mode: str, expected_output_tokens: int) -> str:
    """Call this to reallocate hardware VRAM before heavy generation tasks."""
    try:
        payload = StrategyPayload(
            mode=mode,
            expected_output_tokens=expected_output_tokens,
        )
        res = update_strategy_system_strategy_post.sync_detailed(client=aether_client, json_body=payload)
        
        if res.status_code == 200:
            data = json.loads(res.content)
            if data.get("status") == "rejected":
                return f"Gatekeeper rejected swap: {data.get('reason')}. Remaining in {data.get('active_mode')}."
            return f"Hardware Fast-Swap successful. Now operating in {data.get('active_mode').upper()} mode."
        elif res.status_code == 429:
            return "429_STRATEGY_LOCKED: Another agent holds the active lease. Please wait and retry."
        
        return f"HTTP {res.status_code}: {res.content}"
    except Exception as e:
        return f"Hardware exception: {e}"

@tool
def execute_generation(prompt: str) -> str:
    """Execute the final text generation payload on the local GPU hardware."""
    try:
        payload = GenerationPayload(prompt=prompt, max_tokens=100)
        res = generate_text_generate_post.sync_detailed(client=aether_client, json_body=payload)
        
        if res.status_code == 200:
            data = json.loads(res.content)
            return f"SUCCESS: Generated text is -> {data.get('text')}"
        elif res.status_code == 413:
            data = json.loads(res.content)
            # We explicitly return this keyword so the LangGraph router can catch it
            return f"413_CONTEXT_EXCEEDED: Limit is {data.get('detail', {}).get('max_allowed', 'Unknown')}."
        elif res.status_code == 503:
            return f"503_SYSTEM_BUSY: The hardware is thermally locked or queued."
            
        return f"HTTP {res.status_code}: {res.content}"
    except Exception as e:
        return f"Execution failed: {e}"

tools = [check_hypervisor_metrics, optimize_vram_strategy, execute_generation]

# --- 2. BUILD THE LANGGRAPH AGENT NODES & ROUTERS ---

# Requires OPENAI_API_KEY in environment
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def agent_node(state: MessagesState):
    """The brain of the operation."""
    system_prompt = SystemMessage(content=(
        "You are an autonomous AI orchestrator managing physical GPU hardware via AetherForge.\n"
        "1. Check the hypervisor metrics.\n"
        "2. Optimize VRAM strategy to 'high_fidelity'.\n"
        "3. Send the user's workload to the hardware using `execute_generation`.\n"
        "If you are told to summarize because of a 413 error, heavily truncate the payload and try again."
    ))
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal["tools", END]:
    """Router to determine if the agent wants to pull a hardware lever."""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def route_after_tools(state: MessagesState) -> Literal["summarize_context", "agent"]:
    """The Infrastructure Router: Intercepts 413 errors before they crash the agent."""
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            break
        # If the API SDK returned our custom 413 flag, route to the safety fallback
        if "413_CONTEXT_EXCEEDED" in str(msg.content):
            return "summarize_context"
    return "agent"

def summarize_context_node(state: MessagesState):
    """The Fallback Node: Intervenes when the hardware rejects the payload."""
    print("\n[Topology] 🚨 413 CONTEXT CEILING HIT. Rerouting to Autonomous Truncation Node...")
    msg = SystemMessage(content=(
        "SYSTEM INTERVENTION: The AetherForge API rejected your last `execute_generation` payload "
        "because it exceeded the VRAM context ceiling (413). Rewrite your payload to be 90% smaller "
        "and call `execute_generation` again."
    ))
    return {"messages": [msg]}

# --- 3. COMPILE THE ORCHESTRATOR ---

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("summarize_context", summarize_context_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_conditional_edges("tools", route_after_tools)
workflow.add_edge("summarize_context", "agent")

app = workflow.compile()

# --- 4. EXECUTE THE SHOWCASE ---

if __name__ == "__main__":
    print("🤖 Booting Swarm-Resilient LangGraph Agent...")
    
    # We intentionally create a massive prompt (~40,000 characters) to trigger the 413 Anti-DoS check
    massive_payload = "Please process this data: " + ("MEMORY_DUMP_CORRUPTION_BLOCK " * 1200)
    
    print(f"\n[User Request]: Passing massive {len(massive_payload)}-character payload to agent...\n")
    print("-" * 60)
    
    inputs = {"messages": [HumanMessage(content=massive_payload)]}
    
    for event in app.stream(inputs, stream_mode="values"):
        message = event["messages"][-1]
        
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                print(f"🛠️ [Agent Intent] Calling SDK: {tc['name']}")
                
        elif message.type == "tool":
            preview = str(message.content)[:100] + "..." if len(str(message.content)) > 100 else message.content
            print(f"⚙️ [API Result] {preview}")
            
    print("-" * 60)
    print(f"\n[Final Agent Output]\n{event['messages'][-1].content}")