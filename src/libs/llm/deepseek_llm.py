"""DeepSeek LLM provider implementation."""

from __future__ import annotations

from libs.llm.openai_llm import OpenAICompatibleLLM


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek chat-completions provider implementation."""

    api_url_env_var = "DEEPSEEK_API_URL"
    api_key_env_var = "DEEPSEEK_API_KEY"
    default_base_url = "https://api.deepseek.com/chat/completions"
    provider_label = "deepseek"
