#Constants
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 3000 # Adjust as needed
CHUNK_OVERLAP = 500 # Adjust as needed
RETRIEVER_K = 5 # Number of chunks to retrieve, adjust as needed
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL_NAME = "deepseek-r1:latest" # Default model name, can be any open-source model
LLM_TEMPERATURE = 0.1 # Temperature for LLM generation