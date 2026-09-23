"""Ollama API Client targeting local models running on Fedora PC worker / localhost tunnel."""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

_local_models_src = Path(__file__).resolve().parents[4] / "local-models" / "src"
if _local_models_src.exists() and str(_local_models_src) not in sys.path:
    sys.path.insert(0, str(_local_models_src))
try:
    from local_models.telemetry import TelemetryLogger, UsageEvent
except ImportError:  # Keep remote jury work usable when sibling tools are absent.
    TelemetryLogger = None  # type: ignore[assignment,misc]
    UsageEvent = None  # type: ignore[assignment,misc]


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
            ]
            self.candidate_urls = [u.rstrip("/") for u in self.candidate_urls if u]
            self.base_url = self._resolve_active_url()
        self.timeout = timeout
        self.telemetry = TelemetryLogger() if TelemetryLogger else None

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
        task: str = "ollama_chat",
    ) -> str:
        """Generate a chat response, recording metadata without prompt content."""
        request_id = str(uuid.uuid4())
        prompt_chars = sum(len(message.get("content", "")) for message in messages)
        if self.telemetry:
            self.telemetry.start_request(request_id, "linear-a", task, prompt_chars=prompt_chars)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        t = timeout or self.timeout
        execution = "local" if "localhost" in self.base_url or "127.0.0.1" in self.base_url else "remote"
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=t) as client:
                res = client.post(f"{self.base_url}/api/chat", json=payload)
                res.raise_for_status()
                data = res.json()
            elapsed = (time.perf_counter() - started) * 1000
            self._record(request_id, task, model, execution, "success", elapsed)
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            self._record(request_id, task, model, execution, "failure", elapsed, exc)
            raise

    def _record(
        self,
        request_id: str,
        task: str,
        model: str,
        execution: str,
        status: str,
        latency_ms: float,
        error: Optional[BaseException] = None,
    ) -> None:
        """Telemetry is additive; it never changes jury routing or error handling."""
        if not self.telemetry or not UsageEvent:
            return
        self.telemetry.log_attempt(
            request_id,
            1,
            provider="ollama",
            model=model,
            execution=execution,
            status=status,
            latency_ms=latency_ms,
            error=error,
        )
        self.telemetry.log(
            UsageEvent(
                project="linear-a",
                task=task,
                provider="ollama",
                model=model,
                execution=execution,
                latency_ms=latency_ms,
                status=status,
                error=type(error).__name__ if isinstance(error, BaseException) else error,
                request_id=request_id,
            )
        )
