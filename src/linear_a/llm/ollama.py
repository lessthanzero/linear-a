"""Ollama API Client targeting local models running on Fedora PC worker / localhost tunnel."""

import os
from typing import Any, Dict, List, Optional
import httpx


class OllamaClient:
    """HTTP Client for Ollama models (gemma2:9b, qwen2.5:7b, llama3.2:3b)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        if base_url:
            self.base_url = base_url.rstrip("/")
            self.candidate_urls = [self.base_url]
        else:
            self.candidate_urls = [
                os.getenv("OLLAMA_HOST"),
                "http://localhost:11434",
                "http://127.0.0.1:11434",
                "http://100.103.226.101:11434",
                "http://192.168.1.172:11434",
            ]
            self.candidate_urls = [u.rstrip("/") for u in self.candidate_urls if u]
            self.base_url = self._resolve_active_url()
        self.timeout = timeout

    def _resolve_active_url(self) -> str:
        for url in self.candidate_urls:
            try:
                with httpx.Client(timeout=1.5) as client:
                    res = client.get(f"{url}/api/tags")
                    if res.status_code == 200:
                        return url
            except Exception:
                continue
        return self.candidate_urls[0]

    def is_available(self, timeout: float = 10.0) -> bool:
        """Check if remote Ollama instance is online and responding."""
        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models on remote Ollama instance."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m["name"] for m in data.get("models", [])]
                return []
        except Exception:
            return []

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "qwen2.5:7b",
        temperature: float = 0.1,
        timeout: Optional[float] = None,
    ) -> str:
        """Generate a chat response from the remote Ollama model."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        t = timeout or self.timeout
        with httpx.Client(timeout=t) as client:
            res = client.post(f"{self.base_url}/api/chat", json=payload)
            res.raise_for_status()
            data = res.json()
            return data.get("message", {}).get("content", "")
