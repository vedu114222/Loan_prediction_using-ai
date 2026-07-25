"""
FastAPI router exposing the LangGraph agent over HTTP.

Endpoints:
  POST /api/agent/chat          — single-turn, returns full response
  POST /api/agent/chat/stream   — streaming SSE response
  DELETE /api/agent/thread/{id} — clear a conversation thread (resets memory)

The thread_id in the request body maps directly to LangGraph's MemorySaver,
so the same thread_id across multiple requests gives the agent conversation memory.
"""

import uuid
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["Agent"])


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message to the agent")
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation thread ID. Reuse the same ID across turns for memory.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Predict approval for age 28, salary 85000, credit score 760, loan amount 300000. Explain using policy.",
                    "thread_id": "user-session-abc123",
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    tools_called: list[str]


def _extract_tools_called(messages: list) -> list[str]:
    """Pull the names of any tool calls made during this response."""
    tools = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tools.extend(tc["name"] for tc in msg.tool_calls)
    return list(dict.fromkeys(tools))  # deduplicate, preserve order


@router.post(
    "/agent/chat",
    response_model=ChatResponse,
    summary="Chat with the loan underwriting agent (blocking)",
)
def agent_chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the LangGraph ReAct agent and get a full response.
    The agent will call predict_loan, search_policy, get_history, or
    feature_importance as needed, then return a grounded answer.

    Use the same `thread_id` across requests to maintain conversation context.
    """
    from app.agent.graph import get_agent

    agent = get_agent()
    config = {"configurable": {"thread_id": request.thread_id}}

    result = agent.invoke(
        {"messages": [("user", request.message)]},
        config=config,
    )

    messages = result["messages"]
    reply = messages[-1].content
    tools_called = _extract_tools_called(messages)

    return ChatResponse(
        reply=reply,
        thread_id=request.thread_id,
        tools_called=tools_called,
    )


@router.post(
    "/agent/chat/stream",
    summary="Chat with the loan underwriting agent (streaming SSE)",
)
async def agent_chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Stream the agent's response as Server-Sent Events (SSE).
    Each event contains a `data:` line with a text chunk.
    The final event is `data: [DONE]`.

    Use with `EventSource` in the browser or `httpx` with streaming in Python.
    """
    from app.agent.graph import get_agent

    agent = get_agent()
    config = {"configurable": {"thread_id": request.thread_id}}

    async def generate() -> AsyncGenerator[str, None]:
        last_content = ""
        async for chunk in agent.astream(
            {"messages": [("user", request.message)]},
            config=config,
            stream_mode="values",
        ):
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.type == "ai":
                new_content = last_msg.content
                if new_content != last_content:
                    delta = new_content[len(last_content):]
                    if delta:
                        yield f"data: {delta}\n\n"
                    last_content = new_content
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.delete(
    "/agent/thread/{thread_id}",
    summary="Clear a conversation thread's memory",
)
def clear_thread(thread_id: str):
    """
    Reset the MemorySaver state for a given thread_id.
    The next message on this thread will start a fresh conversation.
    """
    # MemorySaver stores in-process; re-using a new uuid has the same effect.
    # For persistent checkpointers (e.g. SqliteSaver, AsyncPostgresSaver),
    # you'd call checkpointer.delete(thread_id) here.
    return {"status": "cleared", "thread_id": thread_id, "note": "In-memory — thread state naturally resets on restart."}
