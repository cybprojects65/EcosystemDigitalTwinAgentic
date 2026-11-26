# Massaciuccoli Conversational Ecosystem Baseline

A domain-specific environmental chatbot for the Massaciuccoli Lake basin, Tuscany, Italy. The application combines scientific documents, georeferenced ecosystem records, and remotely retrieved observations with a large language model (LLM).

This project implements the **agentic baseline** in the study-specific sense of an LLM-led conversational architecture: the LLM interprets retrieved evidence and constructs an answer. It does not autonomously execute scientific tools or run the query-specific analytical modules of the reactive EcosystemDigitalTwin framework.

## How to cite:

Coro, G. and Bove, B. 2026. Building Conversational Digital Twins of Natural Ecosystems with Reactive Artificial Intelligence.

## Features

- Scientific context from twelve PDF documents and an ecosystem CSV.
- Persistent Chroma indexing with `nomic-embed-text` embeddings.
- Exact cosine-similarity retrieval with a **0.60 minimum threshold** and no fixed result count.
- Independent requests without conversation history or server-side chat sessions.
- Conditional retrieval of updated observations using ten reference queries and a separate **0.50 threshold**.
- A Flask web interface and JSON chat endpoint.
- Local Llama 3.2 3B by default; configurable LLM endpoint, including the D4Science Ollama-compatible Gemini bridge.

## Project structure

| Path | Purpose |
| --- | --- |
| `dockerproject/app.py` | Indexing, retrieval, prompts, live observations, and Flask routes |
| `dockerproject/data/` | Twelve scientific PDFs and the ecosystem CSV |
| `dockerproject/templates/index.html` | Browser interface |
| `dockerproject/Dockerfile` | Container image and Python dependencies |
| `dockerproject/start.sh` | Starts local Ollama, downloads models, and launches Flask |
| `run_langchain_chroma.txt` | Earlier Docker command examples |
| `README_revision.md` | Changes and validation notes for the revised baseline |

The supplied archive also contains a Windows `.bat` launcher at its outer root. Its image name follows the earlier Docker instructions; use the commands below when building the image with the name used in this README.

## Quick start with Docker

Prerequisites: Docker Engine or Docker Desktop, internet access for installation and model downloads, and sufficient memory and disk space for the models, documents, and index. A GPU is optional.

From this project's root directory:

```bash
cd dockerproject
docker build -t massaciuccoli-baseline .
docker run --name massaciuccoli-baseline \
  -p 127.0.0.1:5000:5000 \
  -v massaciuccoli_ollama:/root/.ollama \
  -v massaciuccoli_chroma:/app/chroma_db \
  massaciuccoli-baseline
```

Open [http://localhost:5000](http://localhost:5000).

For an NVIDIA-enabled Docker installation, add `--gpus all` to `docker run`. Without that option, local inference can run on the CPU.

At startup, the container starts Ollama, pulls `llama3.2:3b` and `nomic-embed-text`, and launches the application. Initial indexing may take substantial time because all PDF passages must be embedded before Flask starts accepting requests. Subsequent starts reuse a completed index for the same corpus and chunk settings.

The named volumes preserve downloaded models and Chroma data when the container is replaced. To restart the existing container:

```bash
docker start -a massaciuccoli-baseline
```

Flask's built-in server is intended for local experimentation; production deployment requires appropriate serving and access controls.

## Running directly with Python

Use a Python environment compatible with the dependencies listed in `dockerproject/Dockerfile`. The container uses Ubuntu 22.04 and its system Python; package versions are not pinned.

From `dockerproject/`, create and activate a virtual environment, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install flask ollama requests langchain langchain-classic \
  langchain-community langchain-ollama langchain-chroma \
  langchain-text-splitters langchainhub chromadb pysqlite3-binary \
  pandas pypdf scikit-learn numpy
```

The application imports `pysqlite3` explicitly; availability of its binary package depends on platform. The Docker image provides the intended Linux environment.

Start the Ollama service separately and download both models:

```bash
ollama serve
```

In another terminal, with the virtual environment active:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python3 app.py
```

If Ollama is already running as a service, do not start a second instance.

## Model and endpoint configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `LLM_MODEL` | `llama3.2:3b` | Model name sent to the generation service |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama-compatible generation endpoint |
| `EMBEDDING_BASE_URL` | `http://localhost:11434` | Ollama endpoint serving `nomic-embed-text` |

### Gemini through D4Science

When the **D4Science Ollama-compatible bridge to Google Cloud** is active, the LLM name can be changed to `gemini-3.5-flash`:

```bash
export LLM_MODEL=gemini-3.5-flash
export OLLAMA_BASE_URL=http://bridge-host:11435
export EMBEDDING_BASE_URL=http://localhost:11434
python3 app.py
```

Replace `bridge-host:11435` with the actual configured endpoint. The bridge must already be running with the required Google Cloud access and credentials. This project does not implement the bridge or its authentication setup. Keep the embedding endpoint pointed at an Ollama service hosting `nomic-embed-text`.

For Docker, pass the settings as environment variables:

```bash
docker run --name massaciuccoli-gemini \
  -p 127.0.0.1:5000:5000 \
  -e LLM_MODEL=gemini-3.5-flash \
  -e OLLAMA_BASE_URL=http://bridge-host:11435 \
  -e EMBEDDING_BASE_URL=http://localhost:11434 \
  -v massaciuccoli_ollama:/root/.ollama \
  -v massaciuccoli_chroma:/app/chroma_db \
  massaciuccoli-baseline
```

Here `bridge-host` must be reachable from inside the container. `localhost` refers to the container itself. The existing startup script still downloads the default local LLM as well as the embedding model, even when generation is delegated to the bridge.

## Data and indexing

The application reads `.pdf` and `.csv` files from `dte_agentic/data/`. The supplied collection has twelve PDFs and one CSV containing coordinates, ecosystem stressors, and previously calculated risk classifications.

PDF text is split into chunks of up to **1,200 characters**, with no overlap. Each PDF chunk receives a Nomic embedding and is stored in Chroma.

**Every CSV is loaded in full at startup and included in every answer prompt.** CSV headers, descriptions, row order, quoted fields, and all data rows are preserved (a UTF-8 BOM is removed). CSVs are never chunked, embedded, sampled, or filtered by retrieval scores. Restart the app after changing the data files. These files are supplied as text, not as a computational table-analysis tool.

The PDF index fingerprint includes PDF filenames and bytes, chunk settings, and embedding model name. A new index namespace prevents reuse of previous collections containing CSV chunks. The first start after this update rebuilds the PDF index automatically; older collections are retained. CSV changes require a restart but do not trigger PDF reindexing.

## Request processing

1. Receive the current message; previous conversation turns are not included.
2. Compare its embedding against ten fixed expressions about recent observations.
3. If the highest cosine similarity is strictly greater than `LIVE_QUERY_SIMILARITY_THRESHOLD` (default **0.6**), fetch the remote observation table and append all rows, including headers, to the current input.
4. Search the PDF index using the original question, with up to `RAG_TOP_K` candidates (default **8**).
5. Retain PDF chunks with squared-L2 distance at most `RAG_MAX_DISTANCE` (default **0.6**), deduplicate, and order by increasing distance.
6. Supply the selected PDF chunks, **every complete local CSV**, the current input, and fixed behavioural instructions to the LLM. Source filenames label each evidence item.
7. Return the generated answer.

When no PDF chunks qualify, the full CSV evidence is still supplied. PDF retrieval thresholds and candidate limits never limit CSV inclusion.

### Context capacity

Configure `NUM_CTX` and an inference model/backend with sufficient capacity; a larger setting cannot increase the model's supported maximum. The required token count depends on its tokenizer. No automatic CSV truncation or summarisation is performed by the app; a backend may reject or truncate an oversized prompt. This change guarantees full CSV inclusion in the submitted prompt, while complete model processing depends on backend capacity.

Run offline regression checks from `dte_agentic/` with:

```bash
python -m unittest discover -s tests -v
```

The browser can display earlier messages, but they are not model context. Follow-up requests should repeat any information needed to interpret them independently.

### Updated observations

The remote observation-table URL is specified in `fetch_daily_lake_data()` in `app.py`. Retrieval uses a five-second HTTP timeout. HTTP or parsing failures add a note stating that updated observations could not be retrieved and that the local knowledge base is being used.

The function name retains “daily” from the earlier implementation. Actual temporal coverage and aggregation are determined by the remote table, not by this name. The app does not itself acquire raw sensor observations or recompute their aggregates.

## API

`GET /` serves the interface. `POST /chat` accepts a JSON object with a non-empty string field named `message`:

```bash
curl -X POST http://localhost:5000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Which ecosystem variables are available?"}'
```

A successful response is shaped as:

```json
{"response": "Generated answer"}
```

Invalid or empty input returns HTTP 400. Processing failures return HTTP 500 and are logged by Flask.

## Scientific scope and reproducibility

The baseline interprets retrieved environmental evidence and previously generated model outputs. It does not run MaxEnt, calculate dataset-wide correlations, retrain risk models, or execute Random Forest scenario projections. Requests asking for those analyses do not create new numerical evidence through executable scientific tools.

For reproducible experiments, retain the exact source revision, input files, remote-table snapshot, model identifiers and versions, generation settings, dependency versions, and query set. The Dockerfile downloads current packages and model tags rather than pinning immutable versions. Generation parameters and context limits are configured in `app.py` through environment variables, including `NUM_CTX`, `NUM_PREDICT`, `TEMPERATURE`, `TOP_P`, and `SEED`.

The current revision changes the earlier baseline's history handling, chunk size, retrieval policy, and live-row inclusion. Published benchmark scores should describe the configuration actually evaluated; rerun the benchmark if assigning results to this revised configuration.

Validation during editing covered Python syntax and isolated behavioural checks with mocked services: threshold selection, more than three matching chunks, empty retrieval, ranking, live-request routing, all-row inclusion, retrieval failures, independent requests, and invalid input. Full Chroma ingestion, container startup, and live LLM inference were not tested in that environment.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Browser cannot connect immediately | Initial model downloads or corpus indexing may still be running; inspect container logs. |
| LLM or embedding connection fails | Verify the relevant endpoint and that its required model is available. |
| Gemini fails while local embeddings work | Verify bridge reachability, model mapping, and externally configured Google Cloud access. |
| New papers appear absent | Confirm they are in the mounted `data/` directory and restart the app to select the updated collection. |
| Prompt is too large | Inspect the number of threshold matches and the inference service's context capacity. |
| Follow-up answer ignores earlier context | Requests are intentionally independent; include the necessary context in the current message. |
| Live data unavailable | Check network access and the configured remote table; the app reports a fallback to local evidence. |

## Related framework

The reactive ecosystem-modelling framework is available at [EcosystemDigitalTwin](https://github.com/cybprojects65/EcosystemDigitalTwin).
