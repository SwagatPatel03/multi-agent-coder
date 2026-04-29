import os
from typing import TypedDict, cast

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMConfig(BaseModel):
    """Configuration for the NVIDIA LLM API."""

    model: str = "google/gemma-4-31b-it"
    invoke_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    max_tokens: int = 16384
    # Lower temperature is critical for coding/JSON generation tasks
    temperature: float = 0.2
    top_p: float = 0.95


class _ResponseMessage(TypedDict):
    content: str


class _ResponseChoice(TypedDict):
    message: _ResponseMessage


class _ResponsePayload(TypedDict):
    choices: list[_ResponseChoice]


class _RequestMessage(TypedDict):
    role: str
    content: str


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

    # Automatically retry up to 3 times with exponential backoff if the API fails
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(
        self, system_prompt: str, messages: list[_RequestMessage]
    ) -> str:
        """
        Asynchronously generates a response from the model.
        Returns the full text string.
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
            "stream": False,  # Stream is false because agents need the complete
            # JSON/code block
            "chat_template_kwargs": {"enable_thinking": True},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.config.invoke_url, headers=self.headers, json=payload
            )

            response.raise_for_status()  # Raise an exception for 4xx/5xx errors

            data = cast(_ResponsePayload, response.json())
            return data["choices"][0]["message"]["content"]


# Singleton instance to be imported by your agents
gemma = GemmaClient()
