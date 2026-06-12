"""
llm.py
Thin wrapper around the Gemini API so the rest of the pipeline doesn't
need to know about message formats, etc.
"""
import json
import re
import google.generativeai as genai
from config import GEMINI_API_KEY, LLM_MODEL

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(LLM_MODEL)


def call_llm(prompt: str, system: str = None, max_tokens: int = 1024) -> str:
    if system:
        prompt = f"{system}\n\nUser:\n{prompt}"

    response = model.generate_content(prompt)

    return response.text.strip()


def call_llm_json(prompt: str, system: str = None, max_tokens: int = 1024):
    if system is None:
        system = (
            "Return ONLY valid JSON. "
            "No markdown. No explanation."
        )

    raw = call_llm(prompt, system=system, max_tokens=max_tokens)

    cleaned = re.sub(
        r"^```(json)?|```$",
        "",
        raw.strip(),
        flags=re.MULTILINE,
    ).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON:\n{raw}")
