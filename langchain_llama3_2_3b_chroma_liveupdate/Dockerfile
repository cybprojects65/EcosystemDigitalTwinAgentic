# Start from Ubuntu 22.04
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV OLLAMA_HOST=0.0.0.0:11434

# 1. Install system dependencies
# Added: build-essential and python3-dev (needed for Chroma's C++ components)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    zstd \
    ca-certificates \
	python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# 3. Install Python libraries
RUN pip3 install --no-cache-dir \
    flask \
    ollama \
    requests \
    langchain \
    langchain-classic \
    langchain-community \
    langchain-ollama \
    langchain-chroma \
    langchain-text-splitters \
    langchainhub \
    chromadb \
    pysqlite3-binary \
    pandas \
    pypdf \
    fonttools \
    scikit-learn \
    numpy
	
COPY . .

RUN chmod +x start.sh

# Create the data and DB folders so they have the right permissions
RUN mkdir -p /app/data /app/chroma_db

EXPOSE 5000
EXPOSE 11434

CMD ["./start.sh"]
