#Constants
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 3000 # Adjust as needed
CHUNK_OVERLAP = 500 # Adjust as needed
RETRIEVER_K = 5 # Number of chunks to retrieve, adjust as needed
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL_NAME = "deepseek-r1:latest" # Default model name, can be any open-source model
LLM_TEMPERATURE = 0.1 # Temperature for LLM generation

# Silicon Flow API Configuration
SILICON_FLOW_API_KEY = "sk-uyjxppcyvkvfaqyinuvhaelyrsycerybudbknissurqgkbhx"  # Replace with your actual API key
SILICON_FLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICON_FLOW_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"