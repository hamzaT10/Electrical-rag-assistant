from __future__ import annotations

try:
    from openai import OpenAI  # openai>=1.x
except ImportError:
    OpenAI = None
    import openai  # type: ignore[no-redef]

from electrical_rag.core.settings import Settings

PROVIDER_UNAVAILABLE_MESSAGE = (
    "The LLM provider is unavailable. Check the OpenAI-compatible endpoint, "
    "API key, and verify the OpenAI-compatible server is running."
)


class LLMProviderUnavailableError(RuntimeError):
    """Raised when the OpenAI-compatible LLM server cannot respond."""


class OpenAICompatibleClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        if OpenAI is not None:
            self.mode = "v1"
            self.client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
            )
        else:
            self.mode = "legacy"
            openai.api_key = self.settings.llm_api_key
            openai.api_base = self.settings.llm_base_url
            self.client = openai

    def check_health(self) -> tuple[bool, str | None]:
        try:
            if self.mode == "v1":
                self.client.with_options(
                    timeout=self.settings.llm_health_timeout_seconds
                ).models.list()
            else:
                self.client.Model.list()
            return True, None
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _content_to_text(content: object) -> str:
        if isinstance(content, list):
            return " ".join(str(item) for item in content)
        return str(content or "")

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You answer strictly using retrieved context."},
            {"role": "user", "content": prompt},
        ]

    def ask(self, prompt: str) -> str:
        messages = self._messages(prompt)

        if self.mode == "v1":
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    temperature=self.settings.llm_temperature,
                    messages=messages,
                )
            except Exception as exc:
                raise LLMProviderUnavailableError(PROVIDER_UNAVAILABLE_MESSAGE) from exc
            message = response.choices[0].message.content
            return self._content_to_text(message).strip()

        try:
            response = self.client.ChatCompletion.create(  # openai<1.x fallback
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                messages=messages,
            )
        except Exception as exc:
            raise LLMProviderUnavailableError(PROVIDER_UNAVAILABLE_MESSAGE) from exc
        return (response["choices"][0]["message"]["content"] or "").strip()

    def stream(self, prompt: str):
        messages = self._messages(prompt)

        if self.mode == "v1":
            try:
                stream = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    temperature=self.settings.llm_temperature,
                    messages=messages,
                    stream=True,
                )
            except Exception as exc:
                raise LLMProviderUnavailableError(PROVIDER_UNAVAILABLE_MESSAGE) from exc
            for chunk in stream:
                delta = chunk.choices[0].delta
                text = self._content_to_text(getattr(delta, "content", None))
                if text:
                    yield text
            return

        try:
            stream = self.client.ChatCompletion.create(  # openai<1.x fallback
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                messages=messages,
                stream=True,
            )
        except Exception as exc:
            raise LLMProviderUnavailableError(PROVIDER_UNAVAILABLE_MESSAGE) from exc
        for chunk in stream:
            text = chunk["choices"][0].get("delta", {}).get("content", "")
            if text:
                yield text
