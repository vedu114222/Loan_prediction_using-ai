"""
Interactive CLI to chat with the LangGraph loan underwriting agent.

Usage:
    python -m app.agent.run

Each run starts a new conversation thread. Type 'exit' or Ctrl-C to quit.
Type 'new' to start a fresh conversation (clear memory).

Example prompts:
  - Predict approval for age 28, salary 85000, credit score 760, loan amount 300000
  - Why might someone with a 580 credit score get rejected? Check the policy.
  - Show the last 5 predictions and summarise any pattern.
  - Which factor does the model weight most heavily?
"""

import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

from app.agent.graph import get_agent  # noqa: E402


def run_cli():
    print("\n" + "=" * 60)
    print("  Loan Underwriting Agent  (LangGraph + ReAct)")
    print("  Commands: 'exit' to quit, 'new' to reset conversation")
    print("=" * 60 + "\n")

    agent = get_agent()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"[session: {thread_id[:8]}...]\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Goodbye]")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("[Goodbye]")
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print(f"[New session: {thread_id[:8]}...]\n")
            continue

        print("\nAgent: ", end="", flush=True)

        try:
            # Stream token-by-token for a responsive feel
            for chunk in agent.stream(
                {"messages": [("user", user_input)]},
                config=config,
                stream_mode="values",
            ):
                last_msg = chunk["messages"][-1]
                # Only print AI messages (not tool call/result intermediates)
                if hasattr(last_msg, "content") and last_msg.type == "ai":
                    # Clear the line and reprint the full content so far
                    print(f"\r\033[KAgent: {last_msg.content}", end="", flush=True)

            print("\n")

        except Exception as exc:
            print(f"\n[Error: {exc}]\n")


if __name__ == "__main__":
    run_cli()
