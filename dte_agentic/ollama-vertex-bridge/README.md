# Ollama-to-Vertex AI bridge

A dependency-free Python 3 service that accepts common Ollama HTTP API requests and forwards them to a Gemini model on Google Vertex AI.

## Supported endpoints

- `POST /api/generate`
- `POST /api/chat`
- `GET /api/tags`
- `POST /api/show`
- `GET /api/version`
- `GET /healthz`

Both generation endpoints support Ollama's `stream: true` NDJSON response and `stream: false` single-JSON response. The service maps `top_p`, `top_k`, `seed`, and `stop` from Ollama `options` to Vertex generation settings. Temperature and output length use the fixed profile described below.

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
chmod +x ollama_vertex_bridge_basic_auth.py

export VERTEX_PROJECT_ID="d4science-ai-services-prod"
export VERTEX_LOCATION="global"
export VERTEX_API_ENDPOINT="aiplatform.googleapis.com"
export VERTEX_MODEL_ID="gemini-3.5-flash"
export OLLAMA_MODEL_NAME="gemini-3.5-flash"

./ollama_vertex_bridge_basic_auth.py --host 0.0.0.0 --port 11435
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
      "temperature": 0,
      "top_p": 0.95,
      "num_predict": 1536
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
sudo cp ollama_vertex_bridge_basic_auth.py /opt/ollama-vertex-bridge/
sudo chmod 755 /opt/ollama-vertex-bridge/ollama_vertex_bridge_basic_auth.py

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

## Fixed Gemini generation profile

The supplied environment file and Python defaults use:

- `LOCAL_CONTEXT_WINDOW=4096`: bridge-enforced input plus reserved output budget.
- `DEFAULT_TEMPERATURE=0`: forwarded as Vertex `temperature`.
- `DEFAULT_MAX_OUTPUT_TOKENS=1536`: forwarded as Vertex `maxOutputTokens`.

Both `/api/chat` and `/api/generate` enforce this profile. Client `temperature`,
`num_predict`, and `num_ctx` options do not override it. Other supported options
retain their existing mapping; `top_p` defaults to 0.95 as before.
Administrators can change the profile through these environment variables.

Vertex does not implement Ollama's `num_ctx`. Before generation the bridge calls
Vertex `countTokens` for the converted input, including system instructions.
Input tokens plus the reserved 1536 output tokens must not exceed 4096. Thus the
default input allowance is 2560 tokens. Oversized requests receive HTTP 400;
no text, evidence, or conversation messages are silently truncated. Token-count
errors fail the request rather than bypassing the budget. This is a local request
budget, not a restriction of Gemini's native context capacity or an exact replica
of Ollama context handling. The extra API call adds latency and requires countTokens
access for the deployed model. Gemini/provider reasoning and token accounting can
also affect the amount of visible answer text within the output limit.

### Input limits and rejection behaviour

**4096 is the combined input-plus-reserved-output budget, not the input limit.**

| Setting | Default | Meaning |
| --- | ---: | --- |
| Local total budget | 4096 tokens | Input plus reserved maximum output |
| Reserved maximum output | 1536 tokens | Upper bound, not a target answer length |
| Available input budget | 2560 tokens | 4096 minus 1536 |
| Temperature | 0 | Fixed value forwarded to Vertex |

The bridge counts the converted request with Vertex's tokenizer, including system
instructions, supplied evidence, user messages, and any included conversation
history or images. It does not measure input length by characters or words.

- Input of 2560 tokens: accepted by the local budget check.
- Input of 2561 tokens: rejected with HTTP 400 before generation.
- Input of 4096 tokens: also rejected, because no room remains for the reserved output.

The full output allowance is reserved even when an answer might end early.
Oversized input is never automatically truncated, summarised, or reduced by
removing RAG passages. The client must reduce the input and resubmit it. Other
provider errors can still prevent a request that passes this budget check.

Example error message:

```text
Local context budget exceeded: 2561 input tokens + 1536 reserved output tokens > 4096. Reduce the supplied context; no input was truncated.
```

If you instead want **4096 input tokens plus 1536 output tokens**, set the
administrator environment variable `LOCAL_CONTEXT_WINDOW=5632` and restart the
service. Changing only the client's `options.num_ctx` does not change the budget.

Implementation: `Config.__init__()` sets the limits;
`enforce_context_budget()` counts and checks the request; `handle_generation()`
calls this check before requesting generation for both API endpoints. The policy
does not resize Gemini's native context window. Output can still end at the
configured generation limit; input rejection and output-length limiting are
separate behaviours.

After replacing an installed script and environment file, restart the service:

```bash
sudo systemctl restart ollama-vertex-bridge
```

The existing unit may refer to `ollama_vertex_bridge.py`; either install the updated
script at that existing path or update `ExecStart` to the supplied filename and run
`sudo systemctl daemon-reload` before restarting. Authentication settings are unchanged.

Tests: `python3 -m unittest discover -s tests -v` (mocked, no cloud calls).
