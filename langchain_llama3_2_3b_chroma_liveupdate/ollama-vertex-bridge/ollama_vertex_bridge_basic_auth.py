#!/usr/bin/env python3
"""Ollama-compatible HTTP bridge for Google Vertex AI Gemini.

The service accepts the most common Ollama REST requests and translates them
into Vertex AI generateContent/streamGenerateContent requests.

External runtime requirements:
  * gcloud authenticated so `gcloud auth print-access-token` succeeds
  * curl

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable

LOG = logging.getLogger("ollama-vertex-bridge")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


class Config:
    def __init__(self) -> None:
        self.project_id = os.getenv("VERTEX_PROJECT_ID", "d4science-ai-services-prod")
        self.location = os.getenv("VERTEX_LOCATION", "global")
        self.api_endpoint = os.getenv("VERTEX_API_ENDPOINT", "aiplatform.googleapis.com")
        self.vertex_model = os.getenv("VERTEX_MODEL_ID", "gemini-3.5-flash")
        self.ollama_model = os.getenv("OLLAMA_MODEL_NAME", self.vertex_model)
        self.host = os.getenv("BRIDGE_HOST", "0.0.0.0")
        self.port = env_int("BRIDGE_PORT", 11435)
        self.timeout = env_int("VERTEX_TIMEOUT_SECONDS", 600)
        self.default_max_tokens = env_int("DEFAULT_MAX_OUTPUT_TOKENS", 2024)
        self.default_temperature = env_float("DEFAULT_TEMPERATURE", 0.5)
        self.default_top_p = env_float("DEFAULT_TOP_P", 0.95)
        self.use_vertex_stream_endpoint = os.getenv("USE_VERTEX_STREAM_ENDPOINT", "true").lower() in {
            "1", "true", "yes", "on"
        }
        self.bridge_username = os.getenv("BRIDGE_USERNAME", "")
        self.bridge_password = os.getenv("BRIDGE_PASSWORD", "")

        if bool(self.bridge_username) != bool(self.bridge_password):
            raise RuntimeError(
                "BRIDGE_USERNAME and BRIDGE_PASSWORD must either both be set or both be empty"
            )

    @property
    def vertex_method(self) -> str:
        return "streamGenerateContent" if self.use_vertex_stream_endpoint else "generateContent"

    @property
    def vertex_url(self) -> str:
        return (
            f"https://{self.api_endpoint}/v1/projects/{self.project_id}"
            f"/locations/{self.location}/publishers/google/models/"
            f"{self.vertex_model}:{self.vertex_method}"
        )


CONFIG = Config()


class BridgeError(Exception):
    def __init__(self, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


def get_access_token() -> str:
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise BridgeError("gcloud is not installed or is not in PATH", 500) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "gcloud authentication failed").strip()
        raise BridgeError(f"Unable to obtain Google access token: {detail}", 502) from exc
    except subprocess.TimeoutExpired as exc:
        raise BridgeError("Timed out while obtaining Google access token", 504) from exc

    token = result.stdout.strip()
    if not token:
        raise BridgeError("gcloud returned an empty access token", 502)
    return token


def run_vertex_request(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Call Vertex through curl and return response objects plus duration in ns."""
    token = get_access_token()
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--max-time",
        str(CONFIG.timeout),
        "-X",
        "POST",
        "-H",
        f"Authorization: Bearer {token}",
        "-H",
        "Content-Type: application/json",
        CONFIG.vertex_url,
        "--data-binary",
        "@-",
    ]

    started = time.perf_counter_ns()
    try:
        result = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            timeout=CONFIG.timeout + 15,
        )
    except FileNotFoundError as exc:
        raise BridgeError("curl is not installed or is not in PATH", 500) from exc
    except subprocess.TimeoutExpired as exc:
        raise BridgeError("Vertex AI request timed out", 504) from exc
    duration = time.perf_counter_ns() - started

    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()

    if result.returncode != 0:
        detail = stdout or stderr or f"curl exited with status {result.returncode}"
        try:
            error_obj = json.loads(stdout)
            detail = error_obj.get("error", {}).get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise BridgeError(f"Vertex AI request failed: {detail}", 502)

    if not stdout:
        raise BridgeError("Vertex AI returned an empty response", 502)

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BridgeError(f"Vertex AI returned invalid JSON: {exc}", 502) from exc

    # streamGenerateContent normally returns a JSON array; generateContent returns one object.
    if isinstance(parsed, list):
        responses = [item for item in parsed if isinstance(item, dict)]
    elif isinstance(parsed, dict):
        responses = [parsed]
    else:
        raise BridgeError("Unexpected Vertex AI response type", 502)

    for item in responses:
        if "error" in item:
            message = item.get("error", {}).get("message", "Unknown Vertex AI error")
            raise BridgeError(f"Vertex AI request failed: {message}", 502)

    return responses, duration


def normalize_model(requested: Any) -> str:
    # Ollama clients often send arbitrary local aliases. The bridge routes all of
    # them to the configured Vertex model while echoing the requested name back.
    if isinstance(requested, str) and requested.strip():
        return requested.strip()
    return CONFIG.ollama_model


def ollama_options_to_vertex(body: dict[str, Any]) -> dict[str, Any]:
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    generation: dict[str, Any] = {
        "candidateCount": 1,
        "maxOutputTokens": int(options.get("num_predict", CONFIG.default_max_tokens)),
        "temperature": float(options.get("temperature", CONFIG.default_temperature)),
        "topP": float(options.get("top_p", CONFIG.default_top_p)),
    }

    if "top_k" in options:
        generation["topK"] = int(options["top_k"])
    if "seed" in options:
        generation["seed"] = int(options["seed"])
    if "stop" in options:
        stop = options["stop"]
        generation["stopSequences"] = stop if isinstance(stop, list) else [str(stop)]

    fmt = body.get("format")
    if fmt == "json":
        generation["responseMimeType"] = "application/json"
    elif isinstance(fmt, dict):
        generation["responseMimeType"] = "application/json"
        generation["responseSchema"] = fmt

    return generation


def images_to_parts(images: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if not isinstance(images, list):
        return parts
    for image in images:
        if isinstance(image, str) and image:
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": image}})
    return parts


def generate_to_vertex(body: dict[str, Any]) -> dict[str, Any]:
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        raise BridgeError("field 'prompt' must be a string", 400)

    parts: list[dict[str, Any]] = [{"text": prompt}]
    parts.extend(images_to_parts(body.get("images")))
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": ollama_options_to_vertex(body),
    }

    system = body.get("system")
    if isinstance(system, str) and system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    return payload


def map_role(role: str) -> str:
    if role == "assistant":
        return "model"
    # Vertex accepts user/model in conversation contents. Tool messages are
    # represented as user text unless native tool declarations are implemented.
    return "user"


def chat_to_vertex(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise BridgeError("field 'messages' must be an array", 400)

    contents: list[dict[str, Any]] = []
    system_fragments: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = item.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        if role == "system":
            system_fragments.append(content)
            continue

        parts: list[dict[str, Any]] = [{"text": content}]
        parts.extend(images_to_parts(item.get("images")))
        contents.append({"role": map_role(role), "parts": parts})

    if not contents:
        raise BridgeError("no user or assistant messages were supplied", 400)

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": ollama_options_to_vertex(body),
    }
    if system_fragments:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_fragments)}]}
    return payload


def extract_vertex_result(responses: list[dict[str, Any]]) -> tuple[list[str], str, dict[str, int]]:
    chunks: list[str] = []
    finish_reason = "stop"
    usage = {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}

    for response in responses:
        metadata = response.get("usageMetadata")
        if isinstance(metadata, dict):
            for key in usage:
                if isinstance(metadata.get(key), int):
                    usage[key] = metadata[key]

        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            continue
        if candidate.get("finishReason"):
            finish_reason = str(candidate["finishReason"]).lower()
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        text = "".join(
            str(part.get("text", ""))
            for part in parts
            if isinstance(part, dict) and part.get("text") is not None
        )
        if text:
            chunks.append(text)

    return chunks, finish_reason, usage


def split_text_for_streaming(text: str, target_size: int = 80) -> Iterable[str]:
    """Yield readable chunks while preserving the exact concatenated text."""
    if not text:
        return
    start = 0
    length = len(text)
    while start < length:
        end = min(start + target_size, length)
        if end < length:
            newline = text.rfind("\n", start + 1, end + 1)
            space = text.rfind(" ", start + 1, end + 1)
            boundary = max(newline, space)
            if boundary > start:
                end = boundary + 1
        yield text[start:end]
        start = end


def ollama_done_reason(vertex_reason: str) -> str:
    mapping = {
        "max_tokens": "length",
        "max_output_tokens": "length",
        "stop": "stop",
        "finish_reason_unspecified": "stop",
    }
    return mapping.get(vertex_reason, vertex_reason or "stop")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Ollama/0.0 VertexBridge"

    def authentication_enabled(self) -> bool:
        return bool(CONFIG.bridge_username and CONFIG.bridge_password)

    def is_authenticated(self) -> bool:
        if not self.authentication_enabled():
            return True

        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False

        try:
            decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError):
            return False

        return hmac.compare_digest(username, CONFIG.bridge_username) and hmac.compare_digest(
            password, CONFIG.bridge_password
        )

    def require_authentication(self) -> bool:
        if self.is_authenticated():
            return True

        raw = json.dumps({"error": "unauthorized"}).encode("utf-8")
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="Ollama Vertex Bridge"')
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        return False

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def begin_ndjson(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def write_ndjson(self, payload: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        self.wfile.flush()

    def read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise BridgeError("invalid Content-Length", 400) from exc
        if length <= 0:
            return {}
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeError(f"invalid JSON request: {exc}", 400) from exc
        if not isinstance(body, dict):
            raise BridgeError("JSON request body must be an object", 400)
        return body

    def do_GET(self) -> None:  # noqa: N802
        if not self.require_authentication():
            return
        try:
            if self.path in ("/", "/api/version"):
                if self.path == "/":
                    self.send_json(200, {"status": "Ollama is running"})
                else:
                    self.send_json(200, {"version": "0.0.0-vertex-bridge"})
            elif self.path == "/api/tags":
                self.send_json(200, self.tags_response())
            elif self.path in ("/health", "/healthz"):
                self.send_json(200, {"status": "ok", "vertex_model": CONFIG.vertex_model})
            else:
                self.send_json(404, {"error": "not found"})
        except BridgeError as exc:
            self.send_json(exc.status, {"error": str(exc)})
        except Exception:
            LOG.exception("Unhandled GET failure")
            self.send_json(500, {"error": "internal server error"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.require_authentication():
            return
        request_id = uuid.uuid4().hex[:12]
        try:
            body = self.read_json_body()
            LOG.info("request=%s path=%s model=%s stream=%s", request_id, self.path, body.get("model"), body.get("stream", True))

            if self.path == "/api/generate":
                self.handle_generation(body, chat=False)
            elif self.path == "/api/chat":
                self.handle_generation(body, chat=True)
            elif self.path == "/api/show":
                self.send_json(200, self.show_response(body))
            elif self.path in ("/api/pull", "/api/create", "/api/delete", "/api/copy", "/api/push"):
                raise BridgeError("model management is not supported by the Vertex bridge", 501)
            else:
                self.send_json(404, {"error": "not found"})
        except BrokenPipeError:
            LOG.warning("request=%s client disconnected", request_id)
        except BridgeError as exc:
            LOG.warning("request=%s error=%s", request_id, exc)
            self.send_json(exc.status, {"error": str(exc)})
        except Exception:
            LOG.exception("request=%s unhandled POST failure", request_id)
            self.send_json(500, {"error": "internal server error"})

    def handle_generation(self, body: dict[str, Any], chat: bool) -> None:
        requested_model = normalize_model(body.get("model"))
        vertex_payload = chat_to_vertex(body) if chat else generate_to_vertex(body)
        responses, total_duration = run_vertex_request(vertex_payload)
        vertex_chunks, finish_reason, usage = extract_vertex_result(responses)
        text = "".join(vertex_chunks)
        created_at = utc_now()
        stream = body.get("stream", True) is not False

        common_done = {
            "model": requested_model,
            "created_at": utc_now(),
            "done": True,
            "done_reason": ollama_done_reason(finish_reason),
            "total_duration": total_duration,
            "load_duration": 0,
            "prompt_eval_count": usage["promptTokenCount"],
            "prompt_eval_duration": 0,
            "eval_count": usage["candidatesTokenCount"],
            "eval_duration": total_duration,
        }

        if not stream:
            payload = dict(common_done)
            payload["created_at"] = created_at
            if chat:
                payload["message"] = {"role": "assistant", "content": text}
            else:
                payload["response"] = text
                payload["context"] = []
            self.send_json(200, payload)
            return

        self.begin_ndjson()
        for piece in split_text_for_streaming(text):
            if chat:
                chunk = {
                    "model": requested_model,
                    "created_at": utc_now(),
                    "message": {"role": "assistant", "content": piece},
                    "done": False,
                }
            else:
                chunk = {
                    "model": requested_model,
                    "created_at": utc_now(),
                    "response": piece,
                    "done": False,
                }
            self.write_ndjson(chunk)

        final = dict(common_done)
        if chat:
            final["message"] = {"role": "assistant", "content": ""}
        else:
            final["response"] = ""
            final["context"] = []
        self.write_ndjson(final)

    @staticmethod
    def tags_response() -> dict[str, Any]:
        now = utc_now()
        return {
            "models": [
                {
                    "name": CONFIG.ollama_model,
                    "model": CONFIG.ollama_model,
                    "modified_at": now,
                    "size": 0,
                    "digest": "vertex:" + CONFIG.vertex_model,
                    "details": {
                        "parent_model": "",
                        "format": "vertex-ai",
                        "family": "gemini",
                        "families": ["gemini"],
                        "parameter_size": "cloud",
                        "quantization_level": "cloud",
                    },
                }
            ]
        }

    @staticmethod
    def show_response(body: dict[str, Any]) -> dict[str, Any]:
        requested = normalize_model(body.get("model") or body.get("name"))
        return {
            "license": "Google Vertex AI service",
            "modelfile": f"FROM {requested}\n",
            "parameters": (
                f"temperature {CONFIG.default_temperature}\n"
                f"top_p {CONFIG.default_top_p}\n"
                f"num_predict {CONFIG.default_max_tokens}\n"
            ),
            "template": "{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "vertex-ai",
                "family": "gemini",
                "families": ["gemini"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
            "model_info": {
                "general.architecture": "gemini",
                "general.name": CONFIG.vertex_model,
            },
            "capabilities": ["completion", "vision"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama-compatible bridge to Vertex AI Gemini")
    parser.add_argument("--host", default=CONFIG.host, help=f"listen address (default: {CONFIG.host})")
    parser.add_argument("--port", type=int, default=CONFIG.port, help=f"listen port (default: {CONFIG.port})")
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOG.info("Listening on http://%s:%d", args.host, args.port)
    LOG.info("Ollama model alias: %s", CONFIG.ollama_model)
    LOG.info("Vertex target: %s", CONFIG.vertex_url)
    LOG.info("HTTP Basic authentication: %s", "enabled" if CONFIG.bridge_username else "disabled")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
