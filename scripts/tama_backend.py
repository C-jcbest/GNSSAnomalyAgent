"""Bounded local OpenAI-compatible transport for the upstream TAMA workflow.

Only synthetic development plots are sent. Credentials never enter output files.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from BigModel.Base import BigModelBase
from reproduction_contracts import parse_tama_intervals


class LocalBackend(BigModelBase):
    def __init__(self, max_tokens=4096, temperature=0.1, top_p=0.3):
        super().__init__(max_tokens, temperature, top_p)
        self.used_token = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.ledger = Path(os.environ["TAMA_REQUEST_DIR"])
        self.ledger.mkdir(parents=True, exist_ok=True)
        self.budget = int(os.environ.get("TAMA_MAX_REQUESTS", "9"))
        self.model = os.environ["QWEN_MODEL"]

    def text_item(self, content):
        return {"type": "text", "text": content}

    def image_item_from_path(self, image_path):
        return self.image_item_from_base64(self.image_encoder_base64(image_path))

    def image_item_from_base64(self, image_base64):
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}

    def chat(self, message):
        count = len(list(self.ledger.glob("*.started.json")))
        if count >= self.budget:
            raise RuntimeError("TAMA request budget exhausted")
        stem = self.ledger / f"{count + 1:03d}"
        payload = {"model": self.model, "messages": message, "temperature": self.temperature,
                   "top_p": self.top_p, "max_tokens": self.max_tokens,
                   "enable_thinking": False, "response_format": {"type": "json_object"}}
        serial = json.dumps(payload, sort_keys=True).encode("utf-8")
        receipt = {"model": self.model, "request_sha256": hashlib.sha256(serial).hexdigest(),
                   "max_tokens": self.max_tokens, "temperature": self.temperature,
                   "message_count": len(message), "no_retries": True}
        with stem.with_suffix(".started.json").open("x", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2)
        start = time.monotonic()
        endpoint = os.environ["QWEN_BASE_URL"].rstrip("/") + "/chat/completions"
        headers = {"Authorization": "Bearer " + os.environ["QWEN_API_KEY"]}
        try:
            with httpx.Client(timeout=150) as client:
                response = client.post(endpoint, headers=headers, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Provider HTTP {response.status_code}")
            body = response.json()
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise RuntimeError("Model output did not finish normally")
            content = choice["message"]["content"]
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("TAMA requires a JSON object")
            for key in ("abnormal_index", "corrected_abnormal_index"):
                if key in parsed:
                    parse_tama_intervals(parsed[key], int(os.environ["TAMA_SERIES_LENGTH"]))
            usage = body.get("usage", {})
            self.last_used_token = {key: int(usage.get(key, 0)) for key in self.used_token}
            for key, value in self.last_used_token.items():
                self.used_token[key] += value
            receipt.update({"status": "success", "seconds": time.monotonic() - start,
                            "returned_model": body.get("model"), "usage": self.last_used_token,
                            "output": parsed})
            stem.with_suffix(".response.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return content
        except Exception as error:
            receipt.update({"status": "failed", "seconds": time.monotonic() - start,
                            "error_type": type(error).__name__})
            if "parsed" in locals():
                receipt["output"] = parsed
            stem.with_suffix(".response.json").write_text(json.dumps(receipt), encoding="utf-8")
            # Do not expose response bodies, headers, URLs or credentials through upstream logging.
            raise RuntimeError(f"Model request failed; see receipt {count + 1}") from None
