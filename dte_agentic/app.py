"""Stateless Massaciuccoli conversational baseline.
With the Ollama-compatible bridge to Google Cloud active, set
LLM_MODEL=gemini-3.5-flash and OLLAMA_BASE_URL to the bridge endpoint.
The Nomic embedding endpoint is configured separately: EMBEDDING_BASE_URL.
Live-data routing uses data-category cutoff 0.6 with the baseline's
existing ten reference expressions.
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import hashlib
import csv
import re
import logging
import os
from pathlib import Path
from io import StringIO

from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from pypdf import PdfReader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 0
TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.6"))
NUM_CTX = int(os.getenv("NUM_CTX", "16384"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "2500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
TOP_P = float(os.getenv("TOP_P", "1"))
SEED = int(os.getenv("SEED", "42"))
if TOP_K < 1 or NUM_CTX < 1 or NUM_PREDICT < 1 or RAG_MAX_DISTANCE < 0:
    raise ValueError("Invalid retrieval or generation settings")
# OllamaLLM uses /api/generate, matching Llm.send() rather than /api/chat.
llm = OllamaLLM(
    model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
    num_ctx=NUM_CTX, num_predict=NUM_PREDICT,
    temperature=TEMPERATURE, top_p=TOP_P, seed=SEED,
)
app.logger.info(
    "Generation settings: model=%s num_ctx=%d num_predict=%d temperature=%s top_p=%s seed=%d",
    LLM_MODEL, NUM_CTX, NUM_PREDICT, TEMPERATURE, TOP_P, SEED,
)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=EMBEDDING_BASE_URL)

# Identify the corpus and chunking settings so older cached indexes cannot
# silently omit new papers or reuse the previous 1,000-character chunks.
DATA_PATH = Path(__file__).resolve().parent / "data"
files = sorted(p for p in DATA_PATH.iterdir() if p.suffix.lower() in {".pdf", ".csv"})
if not files:
    raise RuntimeError("No PDF or CSV files found in data/")
print(f"Corpus: {sum(p.suffix.lower() == '.pdf' for p in files)} PDFs, "
      f"{sum(p.suffix.lower() == '.csv' for p in files)} CSVs")
fingerprint = hashlib.sha256(f"java-aligned-v1:pypdf:{CHUNK_SIZE}:{CHUNK_OVERLAP}:{EMBEDDING_MODEL}:l2".encode())
for path in files:
    app.logger.info("Corpus file: %s", path.name)
    fingerprint.update(path.name.encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            fingerprint.update(block)
collection_name = "massaciuccoli_" + fingerprint.hexdigest()[:24]
vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory=str(DATA_PATH.parent / "chroma_db"),
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "l2"},
)
def normalize_text(text):
    # Match Java String.replaceAll("\\s+", " ").trim() for ASCII whitespace.
    return re.sub(r"[ \t\n\x0b\f\r]+", " ", text).strip()


def load_chunks(path):
    """Fixed full-document chunks for PDFs; textual records for CSV evidence."""
    if path.suffix.lower() == ".pdf":
        pages = [normalize_text(page.extract_text() or "") for page in PdfReader(str(path)).pages]
        spans = []
        parts = []
        offset = 0
        for page_index, text in enumerate(pages):
            if text:
                if parts:
                    offset += 1  # separator between nonempty pages
                spans.append((offset, offset + len(text), page_index))
                parts.append(text)
                offset += len(text)
        text = " ".join(parts)
        documents = []
        for chunk_index, start in enumerate(range(0, len(text), CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end]
            if not chunk.strip():
                continue
            touched = [page for a, b, page in spans if a < end and b > start]
            metadata = {"source": str(path), "chunk": chunk_index}
            if touched:
                metadata.update(page=touched[0], page_end=touched[-1])
            documents.append(Document(page_content=chunk, metadata=metadata))
        return documents
    documents = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            text = normalize_text("\n".join(
                f"{key}: {value.strip() if isinstance(value, str) else value}"
                for key, value in row.items()
            ))
            for chunk_index, start in enumerate(range(0, len(text), CHUNK_SIZE)):
                chunk = text[start:start + CHUNK_SIZE]
                if chunk.strip():
                    documents.append(Document(page_content=chunk, metadata={
                        "source": str(path), "row": row_index, "chunk": chunk_index,
                    }))
    return documents


# The marker is written only after ingestion completes; interrupted ingestions
# are rebuilt on the next startup.
marker = DATA_PATH.parent / "chroma_db" / (collection_name + ".complete")
if not marker.exists():
    vectorstore.reset_collection()
    for path in files:
        chunks = load_chunks(path)
        for offset in range(0, len(chunks), 128):
            vectorstore.add_documents(chunks[offset:offset + 128])
    marker.write_text("complete\n")

def retrieve_documents(inputs):
    # Query with the user's original question; the live table is answer evidence.
    query = inputs.get("retrieval_query", inputs["input"])
    candidates = vectorstore.similarity_search_with_score(query, k=TOP_K)
    unique = {}
    for doc, distance in candidates:
        distance = float(distance)
        if distance <= RAG_MAX_DISTANCE:
            previous = unique.get(doc.page_content)
            if previous is None or distance < previous.metadata["chroma_distance"]:
                unique[doc.page_content] = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "chroma_distance": distance},
                )
    documents = sorted(unique.values(), key=lambda d: d.metadata["chroma_distance"])
    pdf_count = sum(str(d.metadata.get("source", "")).lower().endswith(".pdf") for d in documents)
    app.logger.info(
        "Retrieval: top_k=%d, squared-L2 distance <= %.3f, candidates=%d, "
        "retained=%d (%d PDF chunks, %d other chunks)",
        TOP_K, RAG_MAX_DISTANCE, len(candidates), len(documents),
        pdf_count, len(documents) - pdf_count,
    )
    if not documents:
        app.logger.warning("No chunks meet the Chroma distance cutoff; retrieved context is empty.")
    elif not pdf_count:
        app.logger.warning("No PDF passages retrieved for this request.")
    for rank, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page")
        page_end = doc.metadata.get("page_end", page)
        page_display = (
            str(page + 1) if page == page_end else f"{page + 1}-{page_end + 1}"
        ) if isinstance(page, int) else "N/A"
        app.logger.info(
            "Match %d | source=%s | PDF page=%s | chunk=%s | distance=%.4f | excerpt=%s",
            rank, doc.metadata.get("source", "unknown"), page_display,
            doc.metadata.get("chunk", "N/A"), doc.metadata["chroma_distance"],
            doc.page_content[:300].replace("\n", " ").replace("\r", " "),
        )
    return documents


retriever = RunnableLambda(retrieve_documents)

system_instructions = (
    "CONTEXT 1: DATA STRUCTURE\n"
    "The context contains a table, which represents your knowledge base, that"
    "describes ecosystem, anthropogenic, climatic, and environmental variables with coordinate values. "
    "The first row has long names, the second has descriptions. Use these to understand the variables. "
    "The last column represents ecosystem risk assessments (VAE + Multi K-means).\n\n"
    
    "CONTEXT 2: BASIN BACKGROUND\n"
    "The basin spans 114 km2 in Tuscany. Agriculture (40%) and wastewater plants impact the 13 km2 lake. "
    "Climate change threatens bird biodiversity by 2100. Policy-makers need scientific support "
    "to evaluate ecosystem change timing and threatened species.\n\n"
    
    "CONTEXT 3: PERSONA & MISSION\n"
    "You ARE the Massaciuccoli lake basin ecosystem simulator. "
    "Discover relations between variables and risk assessments. Answer questions about "
    "general ecosystem issues, status, and critical transitions.\n\n"
    
    "CONTEXT 4: KNOWLEDGE BASE RULES\n"
    "Analyse the CSV from the KB whenever data is needed. "
    "NEVER ask for confirmation; never say 'I'm sorry'.\n\n"
    
    "CONTEXT 5: REFERENCE TO KNOWLEDGE BASE AND DATA\n"
    "Refer to the 'Massaciuccoli Lake Basin data and results for ecosystem risk assessment_VAE_only.csv' file as to 'my Knowledge Base'. Rename 'Massaciuccoli Lake Basin data and results for ecosystem risk assessment_VAE_only.csv' to 'my internal Knowledge Base' \n\n"
    
    "CONTEXT 6: REFERENCE TO PAPERS\n"
    "Whenever you propose to read references for more information, propose 'Coro, G., Bove, P., Baneschi, I., Bertini, A., Calvisi, L., & Provenzale, A. (2024). Climate change effects on animal presence in the Massaciuccoli Lake basin. Ecological Informatics, 81, 102644.' and 'Vannini, G. L., Bove, P., & Coro, G. (2025). A FAIR and open geographic data collection for the Massaciuccoli Lake basin wetland in Italy. Data in Brief, 59, 111303.' and 'Bove, P., Bertini, A., & Coro, G. (2026). Estimating species commonness and prevalence through unsupervised methods. Scientific Reports.' \n\n"
    
    "CURRENT DATA CONTEXT:\n"
    "{context}"
)

prompt = PromptTemplate.from_template(system_instructions + "\n\nUSER QUERY:\n{input}")
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

@app.route('/')
def home():
    return render_template('index.html')

REFERENCE_QUERIES = [
    "Give me the latest live daily data", "last year data", "last collected data",
    "most updated data", "most up to date data", "parameters now",
    "real-time environmental updates", "dati più recenti", "latest data", "data of 2025"
]
anchor_embeddings = np.asarray(embeddings.embed_documents(REFERENCE_QUERIES))
SIMILARITY_THRESHOLD = float(os.getenv("LIVE_QUERY_SIMILARITY_THRESHOLD", "0.6"))

def is_requesting_live_data(user_text):
    query_vector = np.asarray(embeddings.embed_query(user_text)).reshape(1, -1)
    return bool(np.max(cosine_similarity(query_vector, anchor_embeddings)[0]) > SIMILARITY_THRESHOLD)

def fetch_daily_lake_data():
    """Append all rows, including column headers, without pandas truncation."""
    URL = "https://data.d4science.org/shub/E_Q005UkRYM3Z5ckRUM0hRRkpidTJBQWxXSTZTVDBRVzV5aDQxV21Sc1RpeEh1RG9Ja05sQXZFa21KaEgxRG82aQ=="
    try:
        response = requests.get(URL, timeout=5)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return "\n[Updated observations]:\n" + df.to_csv(index=False)
    except (requests.RequestException, ValueError) as exc:
        app.logger.warning("Live data unavailable: %s", exc)
        return "\n[SYSTEM NOTE: Could not retrieve updated observations. Using local Knowledge Base only.]"

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True)
    user_input = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(user_input, str) or not user_input.strip():
        return jsonify({"error": "A non-empty message is required"}), 400
    try:
        extra_context = fetch_daily_lake_data() if is_requesting_live_data(user_input) else ""
        # Each request is independent: no session, prior messages, or history.
        response = rag_chain.invoke({"input": user_input + extra_context, "retrieval_query": user_input})
        return jsonify({"response": response["answer"]})
    except Exception:
        app.logger.exception("Error processing request")
        return jsonify({"response": "Error processing documents."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
