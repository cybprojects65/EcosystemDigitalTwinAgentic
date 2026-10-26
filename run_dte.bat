cd /d "%~dp0dte_agentic"
docker build -t gianpaolocoro/dte_agentic .
if errorlevel 1 exit /b 1

docker run --gpus all -p 5000:5000 -p 11434:11434 -v ollama_models:/root/.ollama -v chroma_db_data_3_2:/app/chroma_db gianpaolocoro/dte_agentic
