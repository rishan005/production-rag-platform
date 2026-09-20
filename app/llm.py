
from openai import OpenAI

from app.config import CONFIG


llm_client = OpenAI(
    api_key=CONFIG.llm_api_key,
    base_url="https://api.groq.com/openai/v1",
)


def call_llm(
    prompt: str,
    max_tokens: int = 300,
    system: str = None
) -> dict:
    """
    Sends a prompt to the Groq LLM using the OpenAI-compatible API.

    Returns:
        dict containing generated text and token usage.
    """

    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = llm_client.chat.completions.create(
        model=CONFIG.llm_model,
        max_tokens=max_tokens,
        messages=messages,
    )

    usage = response.usage

    return {
        "text": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }