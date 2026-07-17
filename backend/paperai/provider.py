from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Settings


class ModelProvider(Protocol):
    model_name: str

    def complete(self, system: str, user: str) -> str: ...


@dataclass
class OpenAICompatibleProvider:
    settings: Settings

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    def complete(self, system: str, user: str) -> str:
        if not self.settings.model_configured:
            raise RuntimeError("No model API key configured. Set PAPERAI_MODEL_API_KEY.")
        body = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
        }
        request = Request(
            f"{self.settings.model_base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.settings.model_api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.model_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"Model API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Model API request failed: {exc.reason}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Model API returned an unexpected response") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Model API returned empty content")
        return content.strip()
