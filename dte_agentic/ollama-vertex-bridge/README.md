# Ollama-to-Vertex AI bridge

A dependency-free Python 3 service that accepts common Ollama HTTP API requests and forwards them to a Gemini model on Google Vertex AI.

## Supported endpoints

- `POST /api/generate`
- `POST /api/chat`
- `GET /api/tags`
- `POST /api/show`
- `GET /api/version`
- `GET /healthz`

Both generation endpoints support Ollama's `stream: true` NDJSON response and `stream: false` single-JSON response. The service maps `temperature`, `top_p`, `top_k`, `num_predict`, `seed`, and `stop` from Ollama `options` to Vertex generation settings.

The bridge routes any requested Ollama model name to `VERTEX_MODEL_ID`, while echoing the requested name in the response. Model installation and deletion endpoints are intentionally unsupported.

## Requirements

- Ubuntu/Linux with Python 3.12 or another recent Python 3
- `curl`
- Google Cloud CLI (`gcloud`)
- Working authentication, verified with:

```bash
gcloud auth print-access-token
```

For a long-running service, authenticate the Linux account used by systemd. Application Default Credentials alone are not sufficient because this implementation intentionally obtains tokens through `gcloud auth print-access-token`.

## Run directly

```bash
chmod +x ollama_vertex_bridge.py

export VERTEX_PROJECT_ID="d4science-ai-services-prod"
export VERTEX_LOCATION="global"
export VERTEX_API_ENDPOINT="aiplatform.googleapis.com"
export VERTEX_MODEL_ID="gemini-3.5-flash"
export OLLAMA_MODEL_NAME="gemini-3.5-flash"

./ollama_vertex_bridge.py --host 0.0.0.0 --port 11435
```

## Test

Non-streaming generation:

```bash
curl http://127.0.0.1:11435/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3.5-flash",
    "prompt": "What is the capital of France?",
    "stream": false,
    "options": {
      "temperature": 0.5,
      "top_p": 0.95,
      "num_predict": 2024
    }
  }'
```

Streaming chat:

```bash
curl -N http://127.0.0.1:11435/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3.5-flash",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": true
  }'
```

List the model:

```bash
curl http://127.0.0.1:11435/api/tags
```

Configure an Ollama client to use:

```text
http://SERVER_ADDRESS:11435
```

For clients that read `OLLAMA_HOST`:

```bash
export OLLAMA_HOST=http://127.0.0.1:11435
```

## Install with systemd

```bash
sudo mkdir -p /opt/ollama-vertex-bridge
sudo cp ollama_vertex_bridge.py /opt/ollama-vertex-bridge/
sudo chmod 755 /opt/ollama-vertex-bridge/ollama_vertex_bridge.py

sudo cp ollama-vertex-bridge.env /etc/default/ollama-vertex-bridge
sudo cp ollama-vertex-bridge.service /etc/systemd/system/
```

Edit `/etc/systemd/system/ollama-vertex-bridge.service` and replace both `CHANGE_ME` values with the Linux user whose `gcloud auth print-access-token` command works. Then run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ollama-vertex-bridge
sudo systemctl status ollama-vertex-bridge
journalctl -u ollama-vertex-bridge -f
```

## Important compatibility notes

This provides wire compatibility for the main text/chat use case, not every Ollama capability. In particular:

- `/api/embed` and `/api/embeddings` are not implemented.
- Native Ollama tool-call response translation is not implemented.
- Model management operations such as pull/create/delete return HTTP 501.
- Vertex's REST streaming endpoint returns a JSON array to `curl`; this bridge parses that response and then emits Ollama NDJSON chunks. The response format is streaming-compatible, but the first Ollama chunk may arrive only after Vertex has completed the upstream request.
- Ollama's opaque `context` token array has no Vertex equivalent, so `/api/generate` returns an empty `context`. Clients should resend prior conversation through `/api/chat` messages.
