# Massaciuccoli Conversational Ecosystem Baseline

A domain-specific environmental chatbot for the Massaciuccoli Lake basin, Tuscany, Italy. The application combines scientific documents, georeferenced ecosystem records, and remotely retrieved observations with a large language model (LLM).

This project implements the **agentic baseline** in the study-specific sense of an LLM-led conversational architecture: the LLM interprets retrieved evidence and constructs an answer. It does not autonomously execute scientific tools or run the query-specific analytical modules of the reactive EcosystemDigitalTwin framework.

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

At startup, the container starts Ollama, pulls `llama3.2:3b` and `nomic-embed-text`, and launches the application. Initial indexing may take substantial time because the CSV records and all PDF pages must be embedded before Flask starts accepting requests. Subsequent starts reuse a completed index for the same corpus and chunk settings.

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

The application reads `.pdf` and `.csv` files from `dockerproject/data/`. The supplied collection has twelve PDFs and one CSV containing coordinates, ecosystem stressors, and previously calculated risk classifications.

PDF pages and CSV records are loaded as documents and split into chunks of up to **1,200 characters**, with **200-character overlap**. Each chunk receives a Nomic embedding and is stored in Chroma. CSV records are indexed as text, not exposed as a computational table-analysis tool.

A content fingerprint includes source filenames and bytes, chunk size, overlap, and embedding model name. Changed files or chunk settings select a new collection. An index-completion marker prevents reuse of interrupted ingestion. Existing older collections are retained. If the underlying embedding model changes while retaining the same name, the fingerprint does not detect that change; regenerate the index in that case.

At startup, indexed embeddings are loaded into memory for exact cosine comparisons. Memory use therefore grows with the corpus.

## Request processing

1. Receive the current message; previous conversation turns are not included.
2. Compare its embedding against ten fixed expressions about recent observations.
3. If the highest cosine similarity is **strictly greater than 0.50**, fetch the remote observation table and append all rows, including headers, as CSV.
4. Embed the resulting input and calculate cosine similarity against every indexed chunk.
5. Select **all chunks with similarity at least 0.60**, ordered by decreasing similarity.
6. Supply the input, selected chunks, and fixed behavioural instructions to the LLM.
7. Return the generated answer.

Retrieval is threshold-based cosine selection, **not maximum marginal relevance (MMR)**. There is no fixed `k` and no candidate-count cutoff. When nothing qualifies, the retrieved context is empty; no below-threshold fallback is injected. The model still receives the current input and fixed instructions.

All qualifying chunks are passed to generation. The application does not limit their combined size, so the inference service must have sufficient context capacity. A backend's context limits may affect processing of large result sets.

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

For reproducible experiments, retain the exact source revision, input files, remote-table snapshot, model identifiers and versions, generation settings, dependency versions, and query set. The Dockerfile downloads current packages and model tags rather than pinning immutable versions. Generation parameters and context limits are not explicitly set by `app.py`; backend defaults apply.

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
