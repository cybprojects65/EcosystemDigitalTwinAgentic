"""Stateless Massaciuccoli conversational baseline.

Retrieval uses all chunks with cosine similarity >= 0.60, not fixed-k MMR.
With the D4Science Ollama-compatible bridge to Google Cloud active, set
LLM_MODEL=gemini-3.5-flash and OLLAMA_BASE_URL to the bridge endpoint.
The Nomic embedding endpoint is configured separately: EMBEDDING_BASE_URL.
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import hashlib
import logging
import os
from pathlib import Path
from io import StringIO

from flask import Flask, render_template, request, jsonify
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
DOCUMENT_SIMILARITY_THRESHOLD = 0.60
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=EMBEDDING_BASE_URL)

# Identify the corpus and chunking settings so older cached indexes cannot
# silently omit new papers or reuse the previous 1,000-character chunks.
DATA_PATH = Path(__file__).resolve().parent / "data"
files = sorted(p for p in DATA_PATH.iterdir() if p.suffix.lower() in {".pdf", ".csv"})
if not files:
    raise RuntimeError("No PDF or CSV files found in data/")
print(f"Corpus: {sum(p.suffix.lower() == '.pdf' for p in files)} PDFs, "
      f"{sum(p.suffix.lower() == '.csv' for p in files)} CSVs")
fingerprint = hashlib.sha256(f"{CHUNK_SIZE}:{CHUNK_OVERLAP}:{EMBEDDING_MODEL}".encode())
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
)
# The marker is written only after ingestion completes; interrupted ingestions
# are rebuilt on the next startup.
marker = DATA_PATH.parent / "chroma_db" / (collection_name + ".complete")
if not marker.exists():
    vectorstore.reset_collection()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for path in files:
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else CSVLoader(str(path))
        chunks = splitter.split_documents(loader.load())
        for offset in range(0, len(chunks), 128):
            vectorstore.add_documents(chunks[offset:offset + 128])
    marker.write_text("complete\n")

# Cache stored embeddings for exact cosine threshold filtering over the entire
# collection. No fixed document count or top-k candidate limit is applied.
indexed = vectorstore.get(include=["documents", "metadatas", "embeddings"])
chunk_vectors = np.asarray(indexed["embeddings"], dtype=np.float32)
if chunk_vectors.size == 0:
    raise RuntimeError("The corpus produced no indexed chunks")

def retrieve_documents(inputs):
    query_vector = np.asarray(embeddings.embed_query(inputs["input"])).reshape(1, -1)
    scores = cosine_similarity(query_vector, chunk_vectors)[0]
    selected = np.flatnonzero(scores >= DOCUMENT_SIMILARITY_THRESHOLD)
    selected = selected[np.argsort(-scores[selected], kind="stable")]
    documents = [
        Document(
            page_content=indexed["documents"][i],
            metadata={
                **(indexed["metadatas"][i] or {}),
                "cosine_similarity": float(scores[i]),
            },
        )
        for i in selected
    ]

    pdf_count = sum(
        str(doc.metadata.get("source", "")).lower().endswith(".pdf")
        for doc in documents
    )
    app.logger.info(
        "Retrieval: %d qualifying chunks (%d PDF chunks, %d other chunks); "
        "cosine threshold >= %.2f",
        len(documents), pdf_count, len(documents) - pdf_count,
        DOCUMENT_SIMILARITY_THRESHOLD,
    )
    if not documents:
        app.logger.warning(
            "No chunks meet the retrieval threshold. No indexed PDF or CSV "
            "passages will be supplied for this request."
        )
    elif not pdf_count:
        app.logger.warning(
            "No PDF passages retrieved for this request; qualifying chunks "
            "come from other indexed sources."
        )

    for rank, doc in enumerate(documents, start=1):
        source = str(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")
        # PyPDFLoader page indices are zero-based; display physical PDF pages
        # starting at 1. These may differ from printed page numbers.
        page_display = (
            page + 1 if source.lower().endswith(".pdf")
            and isinstance(page, int) else "N/A"
        )
        app.logger.info(
            "Match %d | source=%s | PDF page=%s | cosine=%.4f | excerpt=%s",
            rank, source, page_display,
            doc.metadata["cosine_similarity"],
            doc.page_content[:300].replace("\n", " ").replace("\r", " "),
        )

    # This list is passed to the document-combination chain as LLM context.
    # Logs establish passage selection, not whether generation follows it.
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

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    ("human", "{input}"),
])
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
SIMILARITY_THRESHOLD = 0.5

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
        response = rag_chain.invoke({"input": user_input + extra_context})
        return jsonify({"response": response["answer"]})
    except Exception:
        app.logger.exception("Error processing request")
        return jsonify({"response": "Error processing documents."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
