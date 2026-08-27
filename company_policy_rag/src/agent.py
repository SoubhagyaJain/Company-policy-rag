"""
ReAct Agent with vector-index retrieval as a tool and optional session memory.

Architecture note:
- QueryEngineTool wraps the LlamaIndex query engine so the agent can decide
  WHEN to retrieve vs. answer from prior context.
- ChatMemoryBuffer (optional) enables multi-turn follow-ups without external services.
- Retrieval pipeline (Chroma + reranker) is unchanged — memory only affects agent context
  and optional query expansion in the chat layer.

LlamaIndex >= 0.14: ReActAgent is workflow-based — use constructor + run(), not from_tools/chat.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama

from src.citations import (
    begin_citation_turn,
    get_generation_nodes_this_turn,
    select_citations_for_answer,
)
from src.config import settings
from src.language import append_language_hint
from src.indexing import configure_llama_index, get_or_create_index
from src.memory import create_session_memory, trim_memory_to_window
from src.prompts import LOW_CONFIDENCE_MESSAGE, get_agent_system_prompt, resolve_grounding_mode
from src.retriever import build_query_engine
from src.retrieval_trace import build_retrieval_trace
from src.thinking_extract import begin_thinking_turn, get_thinking_this_turn
from src.utils import logger


@dataclass
class AgentTurnResult:
    """Answer plus citations derived from generation nodes (not parallel retrieval)."""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] | None = None
    low_confidence: bool = False
    grounding_mode: str = "balanced"
    thinking: str | None = None
    thinking_events: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: dict[str, Any] | None = None
    retrieval_trace: dict[str, Any] | None = None
    message_id: str | None = None


def configure_llm() -> Ollama:
    """Instantiate Ollama LLM with conservative temperature for factual RAG."""
    return Ollama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        request_timeout=settings.llm_request_timeout,
        context_window=settings.llm_context_window,
    )


def build_policy_search_tool(
    index: VectorStoreIndex | None = None,
    *,
    scope_filters: dict[str, Any] | None = None,
) -> QueryEngineTool:
    """
    Wrap the vector query engine as a named tool for ReAct reasoning.

    Tool description quality directly affects agent routing accuracy — be explicit
    about when to use this tool vs. answering directly.
    """
    query_engine = build_query_engine(index, filters=scope_filters)
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="policy_search",
            description=(
                "Search all indexed PDFs (employee handbook, legal uploads, guides). "
                "Use for any question about content in those documents. "
                "Input should be a clear natural-language query."
            ),
        ),
    )


def extract_agent_response(result: AgentOutput) -> str:
    """Pull assistant text from a workflow AgentOutput."""
    response = result.response
    if response is None:
        return "I could not generate a response."
    content = getattr(response, "content", None)
    if content:
        return str(content).strip()
    return str(response).strip()


def create_agent(
    index: VectorStoreIndex | None = None,
    memory: ChatMemoryBuffer | None = None,
    *,
    scope_filters: dict[str, Any] | None = None,
) -> ReActAgent:
    """
    Build a ReAct agent with policy_search and optional session memory.

    Args:
        index: Vector store index (loads from Chroma if None).
        memory: Per-session ChatMemoryBuffer; created automatically if memory enabled
                and None is passed. Pass the same buffer to chat_with_memory() each turn.
    """
    configure_llama_index()
    Settings.llm = configure_llm()

    idx = index or get_or_create_index()
    tools = [build_policy_search_tool(idx, scope_filters=scope_filters)]

    session_memory = memory
    if session_memory is None and settings.enable_conversation_memory:
        session_memory = create_session_memory()

    agent = ReActAgent(
        name="PolicyAgent",
        description="Company policy and legal document assistant",
        tools=tools,
        llm=Settings.llm,
        system_prompt=get_agent_system_prompt(),
        verbose=settings.agent_verbose,
        streaming=False,
    )
    logger.info(
        "ReAct agent ready | model=%s | tools=%s | memory=%s",
        settings.llm_model,
        [t.metadata.name for t in tools],
        "on" if session_memory else "off",
    )
    return agent


async def chat_with_memory(
    agent: ReActAgent,
    user_message: str,
    memory: ChatMemoryBuffer | None = None,
) -> AgentTurnResult:
    """
    Run one agent turn and trim memory to the configured window.

    LlamaIndex 0.14+: agent.run() persists turns in the provided memory buffer.
    Citations are selected from query-engine source nodes cited in the answer.
    """
    begin_citation_turn()
    begin_thinking_turn()
    handler = agent.run(
        user_msg=append_language_hint(user_message),
        memory=memory,
        max_iterations=settings.agent_max_iterations,
    )
    result = await handler
    answer = extract_agent_response(result)

    citations: list[dict[str, Any]] = []
    if settings.show_citations:
        generation_nodes = get_generation_nodes_this_turn()
        citations = select_citations_for_answer(
            answer,
            generation_nodes,
            user_query=user_message,
        )

    if settings.enable_conversation_memory and memory is not None:
        trim_memory_to_window(memory)

    generation_nodes = get_generation_nodes_this_turn()
    return AgentTurnResult(
        answer=answer,
        citations=citations,
        low_confidence=LOW_CONFIDENCE_MESSAGE in answer,
        grounding_mode=resolve_grounding_mode(),
        thinking=get_thinking_this_turn(),
        retrieval_trace=build_retrieval_trace(generation_nodes),
        message_id=str(uuid.uuid4())[:12],
    )


def run_query(
    agent: ReActAgent,
    user_message: str,
    memory: ChatMemoryBuffer | None = None,
) -> AgentTurnResult:
    """Synchronous single-turn agent query with memory trimming."""
    return asyncio.run(chat_with_memory(agent, user_message, memory=memory))


async def run_query_async(
    agent: ReActAgent,
    user_message: str,
    memory: ChatMemoryBuffer | None = None,
) -> AgentTurnResult:
    """Async wrapper for chat UI integration."""
    return await chat_with_memory(agent, user_message, memory=memory)


def get_agent_context(index: VectorStoreIndex | None = None) -> dict[str, Any]:
    """
    Bundle agent + retriever + memory for a chat session.

    One memory buffer per session — do not share across users.
    """
    from src.retriever import build_retriever

    session_memory = create_session_memory()
    idx = index or get_or_create_index()
    return {
        "agent": create_agent(idx, memory=session_memory),
        "retriever": build_retriever(idx),
        "memory": session_memory,
        "index": idx,
    }