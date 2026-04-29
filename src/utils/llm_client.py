import os
import re
from typing import Literal, TypedDict, cast

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMConfig(BaseModel):
    """Configuration for the NVIDIA LLM API."""

    model: str = "google/gemma-4-31b-it"
    invoke_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    max_tokens: int = 16384
    temperature: float = 0.2
    top_p: float = 0.95


class _ResponseMessage(TypedDict):
    content: str


class _ResponseChoice(TypedDict):
    message: _ResponseMessage


class _ResponsePayload(TypedDict):
    choices: list[_ResponseChoice]


class RequestMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def strip_thinking(text: str) -> str:
    """Removes <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class GemmaClient:
    def __init__(self, config: LLMConfig | None = None):
        if config is None:
            config = LLMConfig()

        self.config = config
        self.api_key = os.getenv("NVIDIA_API_KEY")

        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(self, system_prompt: str, messages: list[RequestMessage]) -> str:
        """
        Asynchronously generates a response from the model.
        Strips any <think>...</think> reasoning blocks before returning.
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages,
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": True},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.config.invoke_url, headers=self.headers, json=payload
            )
            response.raise_for_status()

            data = cast(_ResponsePayload, response.json())
            raw = data["choices"][0]["message"]["content"]
            return strip_thinking(raw)


gemma = GemmaClient()
