"""
LangGraph ReAct agent for loan underwriting assistance.

Architecture:
  User message
      |
      v
  [LLM node] — reasons about which tool(s) to call
      |
      v  (if tool call needed)
  [Tool node] — executes predict_loan / search_policy / get_history / feature_importance
      |
      v
  [LLM node] — synthesises tool results into a grounded answer
      |
      v
  Final response

The graph uses MemorySaver for in-process conversation memory, so the agent
retains context across multiple turns within the same thread_id.

Usage:
    from app.agent.graph import get_agent
    agent = get_agent()

    # Single turn
    result = agent.invoke(
        {"messages": [("user", "Predict approval for age 28, salary 85000...")]},
        config={"configurable": {"thread_id": "session-1"}},
    )
    print(result["messages"][-1].content)

    # Streaming
    for chunk in agent.stream(
        {"messages": [("user", "Explain the credit score policy")]},
        config={"configurable": {"thread_id": "session-1"}},
        stream_mode="values",
    ):
        print(chunk["messages"][-1].content)
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

_SYSTEM_PROMPT = """You are an expert loan underwriting assistant at Apex Lending Group.
You have access to four tools:

1. predict_loan      — Run the ML model on applicant data (age, salary, credit_score, loan_amount)
2. search_policy     — Semantic search over our underwriting policy documents
3. get_history       — Fetch recent loan predictions from the database
4. feature_importance — See which applicant factors the model weights most heavily

Guidelines:
- ALWAYS call search_policy to ground explanations in actual bank policy before drawing conclusions.
- Cite the source document (filename) when quoting policy language.
- When predicting, also call search_policy with a relevant query so you can explain the decision.
- Be concise but thorough. Use bullet points for multi-factor explanations.
- Never invent policy rules — only state what the retrieved documents say.
"""


def _build_llm():
    """
    Build the LLM. Checks environment for available API keys in order:
      1. OpenAI   (OPENAI_API_KEY)
      2. Anthropic (ANTHROPIC_API_KEY)
      3. Google   (GOOGLE_API_KEY)

    Raises a clear error if none are set.
    """
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"[agent] Using OpenAI model: {model}")
        return ChatOpenAI(model=model, temperature=0, streaming=True)

    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        print(f"[agent] Using Anthropic model: {model}")
        return ChatAnthropic(model=model, temperature=0, streaming=True)

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
        print(f"[agent] Using Google model: {model}")
        return ChatGoogleGenerativeAI(model=model, temperature=0)

    raise EnvironmentError(
        "No LLM API key found. Set one of:\n"
        "  OPENAI_API_KEY    -> uses gpt-4o-mini (override with OPENAI_MODEL)\n"
        "  ANTHROPIC_API_KEY -> uses claude-3-5-haiku (override with ANTHROPIC_MODEL)\n"
        "  GOOGLE_API_KEY    -> uses gemini-2.0-flash (override with GOOGLE_MODEL)\n"
        "in your .env file."
    )


@lru_cache(maxsize=1)
def get_agent():
    """
    Build and cache the ReAct agent. Called once per process.

    Returns a compiled LangGraph graph with:
      - ReAct reasoning loop (LLM ↔ tool calls)
      - MemorySaver for multi-turn conversation memory
      - System prompt guiding underwriting behaviour
    """
    from app.agent.tools import ALL_TOOLS  # local import avoids circular at module level

    llm = _build_llm()
    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=_SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent
