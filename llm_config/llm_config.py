import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER_CONFIGS = {
    "groq_cot":             {"type": "groq",   "env": "groq_cot",             "model": "llama-3.3-70b-versatile"},
    "groq_self_consistency":{"type": "groq",   "env": "groq_self_consistency","model": "llama-3.3-70b-versatile"},
    "groq_generation":      {"type": "groq",   "env": "groq_generation",      "model": "llama-3.3-70b-versatile"},
    "groq_free":            {"type": "groq",   "env": "groq_free",            "model": "llama-3.3-70b-versatile"},
    "groq":                 {"type": "groq",   "env": "groq_cot",             "model": "llama-3.3-70b-versatile"},
    "ollama":               {"type": "ollama", "env": None,                   "model": "llama3.1:8b"},
    "openai":               {"type": "openai", "env": "openai",               "model": "gpt-4o-mini"},
    "qwen":                 {"type": "groq",      "env": "qwen",               "model": "qwen/qwen3-32b"},
    "deepseek":             {"type": "deepseek",  "env": "deepseek",           "model": "deepseek-v4-flash"},
    "claude":               {"type": "anthropic", "env": "claude",             "model": "claude-haiku-4-5-20251001"},
}

_client_cache = {}

def _get_client(provider):
    if provider in _client_cache:
        return _client_cache[provider]
    cfg = PROVIDER_CONFIGS[provider]
    t   = cfg["type"]
    if t == "groq":
        from groq import Groq
        key = os.getenv(cfg["env"])
        if not key:
            raise ValueError(f"Provider '{provider}' has no API key configured.")
        client = Groq(api_key=key)
    elif t == "ollama":
        import ollama
        client = ollama.Client()
    elif t in ("openai", "qwen", "deepseek"):
        from openai import OpenAI
        key = os.getenv(cfg["env"])
        if not key:
            raise ValueError(f"Provider '{provider}' has no API key configured.")
        kwargs = {"api_key": key}
        if t == "qwen":
            kwargs["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif t == "deepseek":
            kwargs["base_url"] = "https://api.deepseek.com"
        client = OpenAI(**kwargs)
    elif t == "anthropic":
        import anthropic
        key = os.getenv(cfg["env"])
        if not key:
            raise ValueError(f"Provider '{provider}' has no API key configured.")
        client = anthropic.Anthropic(api_key=key)
    else:
        raise ValueError(f"Unknown provider type '{t}'")
    _client_cache[provider] = client
    return client

def call_llm(prompt: str, provider: str = "groq",
             system: str = None, max_tokens: int = 500) -> str:
    cfg    = PROVIDER_CONFIGS[provider]
    client = _get_client(provider)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if cfg["type"] == "ollama":
        response = client.chat(model=cfg["model"], messages=messages)
        return response["message"]["content"].strip()

    if cfg["type"] == "anthropic":
        kwargs = {"model": cfg["model"], "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
            kwargs["messages"] = [m for m in messages if m["role"] != "system"]
        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()