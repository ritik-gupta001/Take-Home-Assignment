"""
Provider-agnostic wrapper so the rest of the pipeline (extractor.py,
summarizer.py) doesn't need to know whether it's talking to Anthropic or
OpenAI. Selection is controlled by config.LLM_PROVIDER, which auto-detects
based on which API key is present in .env (or set LLM_PROVIDER explicitly).

Two capabilities are exposed:
  - call_structured(): forces JSON-schema-shaped output (used for clause extraction)
  - call_text(): plain text completion (used for summaries)
"""

import json
import logging

from config import LLM_PROVIDER, MODEL_NAME, ANTHROPIC_API_KEY, OPENAI_API_KEY

logger = logging.getLogger(__name__)

if LLM_PROVIDER == "openai":
    import openai
    _client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    import anthropic
    _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

logger.info(f"LLM provider: {LLM_PROVIDER} | model: {MODEL_NAME}")


def call_structured(system_prompt: str, user_prompt: str, tool_schema: dict, max_tokens: int) -> dict:
    """
    Call the LLM with a forced JSON tool/function schema and return the
    parsed structured output as a dict.

    tool_schema should be in Anthropic's tool format:
        {"name": ..., "description": ..., "input_schema": {...}}
    This is translated automatically into OpenAI's function-calling format
    when LLM_PROVIDER == "openai".
    """
    if LLM_PROVIDER == "openai":
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema["description"],
                "parameters": tool_schema["input_schema"],
            },
        }
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[openai_tool],
            tool_choice={"type": "function", "function": {"name": tool_schema["name"]}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)

    else:  # anthropic
        response = _client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system_prompt,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        raise ValueError("No tool_use block returned by the model.")


def call_text(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Call the LLM for a plain text completion (used for summaries)."""
    if LLM_PROVIDER == "openai":
        response = _client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    else:  # anthropic
        response = _client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()
