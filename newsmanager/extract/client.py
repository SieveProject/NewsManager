"""Async Ollama client tuned for sustained batch throughput.

Three settings decide whether a rented GPU is used well or wasted:

`num_ctx`   Ollama's default context is 2048 tokens. An 8,000-char article plus
            this prompt is ~2,600 tokens, so the default would silently drop the
            end of most articles -- no error, just quietly worse extractions.
            It is set explicitly here and validated against the truncation cap.

`keep_alive` Defaults to 5 minutes, after which the model is evicted from VRAM
            and the next request pays a full reload. For a batch run that must
            be indefinite (-1).

`num_parallel` Ollama batches concurrent requests into one forward pass, which is
            where most of the throughput on a rented GPU comes from. Client
            concurrency should match the server's OLLAMA_NUM_PARALLEL; sending
            more just queues, sending fewer leaves the GPU idle.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class GenerationResult:
    ok: bool
    text: str = ""
    parsed: dict | None = None
    error: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "deepseek-r1:14b"
    num_ctx: int = 4096
    num_predict: int = 1024
    temperature: float = 0.0
    # Integer, not "-1": Ollama parses a string as a Go duration and rejects one
    # without a unit ("time: missing unit in duration") -- every request 400'd.
    keep_alive: int = -1
    concurrency: int = 8
    timeout_s: float = 300.0
    retries: int = 3
    # Thinking vem LIGADO por padrão nos modelos que o suportam (R1, v3.1,
    # Qwen3). Numa extração de milhões de artigos a cadeia de raciocínio
    # multiplica os tokens de saída -- que é exatamente o que se paga por hora
    # de GPU -- sem melhorar um preenchimento de schema. Desligado por padrão.
    think: bool = False
    extra_options: dict = field(default_factory=dict)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# DeepSeek-R1 distills reason no matter what `think: false` says: on Ollama
# 0.35.1, deepseek-r1:14b filled `thinking` and returned empty content with
# think=false, with and without a schema -- every token of num_predict spent
# reasoning. Pre-filling an *empty* think block is what makes them answer
# directly; measured: 0 thinking tokens and schema-valid JSON in 62 tokens.
# This is the model's own chat template (`ollama show deepseek-r1:14b
# --template`) for one user turn, sent raw so the prefill survives.
_R1_RAW = "<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n\n</think>\n\n"


def _is_r1(model: str) -> bool:
    return model.split(":", 1)[0].rsplit("/", 1)[-1] == "deepseek-r1"


def _extract_content(data: dict) -> str:
    """Texto da resposta, tolerando /api/chat e /api/generate.

    Mesmo com think=False alguns builds de R1 ainda emitem <think>...</think>
    dentro do content. O bloco é removido aqui para que o JSON sobreviva ao
    parse -- caso contrário todo o lote falharia por erro de decodificação.
    """
    if "message" in data and isinstance(data["message"], dict):
        text = data["message"].get("content", "") or ""
    else:
        text = data.get("response", "") or ""
    if "<think>" in text:
        text = _THINK_BLOCK.sub("", text)
        # Bloco de raciocínio truncado por num_predict: fica sem tag de
        # fechamento, então nada depois dele é aproveitável.
        text = text.split("<think>")[0]
    return text.strip()


class OllamaClient:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrency)
        limits = httpx.Limits(max_connections=cfg.concurrency + 4, max_keepalive_connections=cfg.concurrency + 4)
        self._client = httpx.AsyncClient(base_url=cfg.host, timeout=cfg.timeout_s, limits=limits)

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        r = await self._client.get("/api/tags")
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return {"models": models, "has_model": self.cfg.model in models}

    async def warm(self) -> None:
        """Force the model into VRAM before timing anything.

        The first request after load takes seconds longer; without this the
        benchmark's early samples are dominated by load time and every ETA
        derived from them is wrong.
        """
        await self._client.post(
            "/api/chat",
            json={"model": self.cfg.model, "messages": [{"role": "user", "content": "ok"}],
                  "stream": False, "think": self.cfg.think,
                  "keep_alive": self.cfg.keep_alive,
                  # Sem num_ctx o modelo carrega no contexto padrão do servidor
                  # (32k numa placa de 24 GB) e a primeira requisição real
                  # força um recarregamento -- ou, pior, ele transborda pra CPU.
                  "options": {"num_predict": 1, "num_ctx": self.cfg.num_ctx}},
        )

    def _request(self, prompt: str) -> tuple[str, dict]:
        """Endpoint and base payload for one prompt.

        /api/chat with `think` for models that honour it; for DeepSeek-R1 with
        thinking off, /api/generate in raw mode with an empty think block (see
        _R1_RAW), because R1 ignores `think: false`.
        """
        if _is_r1(self.cfg.model) and not self.cfg.think:
            return "/api/generate", {
                "model": self.cfg.model,
                "prompt": _R1_RAW.format(prompt=prompt),
                "raw": True,
            }
        return "/api/chat", {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "think": self.cfg.think,
        }

    async def generate(self, prompt: str, schema: dict | None = None) -> GenerationResult:
        endpoint, payload = self._request(prompt)
        payload.update({
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.num_predict,
                # Greedy decoding: this is extraction, not generation. Also makes
                # a re-run of the same article reproducible.
                "temperature": self.cfg.temperature,
                **self.cfg.extra_options,
            },
        })
        if schema is not None:
            # Constrained decoding. The model cannot emit malformed JSON, which
            # removes the single largest source of failed rows in batch extraction.
            payload["format"] = schema

        last_err = ""
        t0 = time.time()
        async with self._sem:
            for attempt in range(1, self.cfg.retries + 1):
                try:
                    r = await self._client.post(endpoint, json=payload)
                    if r.status_code >= 500:
                        last_err = f"http {r.status_code}: {r.text[:200]}"
                        await asyncio.sleep(min(2**attempt, 20))
                        continue
                    r.raise_for_status()
                    data = r.json()
                    text = _extract_content(data)
                    try:
                        parsed = json.loads(text) if text.strip() else None
                    except json.JSONDecodeError as e:
                        # Reachable when schema is None, or if the model hit
                        # num_predict mid-object and the JSON was cut off.
                        return GenerationResult(
                            ok=False, text=text, error=f"json decode: {e}",
                            prompt_tokens=data.get("prompt_eval_count", 0),
                            output_tokens=data.get("eval_count", 0),
                            latency_s=time.time() - t0, attempts=attempt,
                        )
                    return GenerationResult(
                        ok=True, text=text, parsed=parsed,
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        output_tokens=data.get("eval_count", 0),
                        latency_s=time.time() - t0, attempts=attempt,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if attempt < self.cfg.retries:
                        await asyncio.sleep(min(2**attempt, 20))
                except httpx.HTTPStatusError as e:
                    # Keep the server's reason: a bare "http 400" hid a bad
                    # keep_alive behind 128 identical failures in the sweep.
                    return GenerationResult(ok=False, error=f"http {e.response.status_code}: {e.response.text[:200]}",
                                            latency_s=time.time() - t0, attempts=attempt)
        return GenerationResult(ok=False, error=last_err or "exhausted retries",
                                latency_s=time.time() - t0, attempts=self.cfg.retries)
