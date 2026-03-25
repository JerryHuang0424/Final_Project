# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) application that enables users to upload PDF documents and chat with them using a local LLM (via Ollama). The application uses Streamlit for the UI, LangChain for the RAG pipeline, and FAISS for vector storage.

## Architecture

The codebase follows a modular structure under the `str/` directory:

- **str/core/**: Core RAG functionality
  - `file_processor.py`: PDF text extraction and chunking using LangChain's RecursiveCharacterTextSplitter
  - `vector_store.py`: FAISS vector store management with GPU support
  - `rag.py`: RAG prompt construction
- **str/models/**: Model interfaces
  - `embeddings.py`: Sentence-transformers embedding model (cached with `@st.cache_resource`)
  - `llm.py`: Ollama client wrapper for streaming chat completions
- **str/ui/**: UI components (not yet fully implemented)
- **str/utils/**: Utility functions (not yet fully implemented)

Main entry point: `app.py` (Streamlit app with sidebar controls)

## Key Configuration (constants.py)

- `EMBEDDING_MODEL`: "sentence-transformers/all-mpnet-base-v2"
- `CHUNK_SIZE`: 3000, `CHUNK_OVERLAP`: 500
- `RETRIEVER_K`: 5 (number of chunks to retrieve)
- `OLLAMA_HOST`: "http://localhost:11434"
- `LLM_MODEL_NAME`: "deepseek-r1:latest"
- `LLM_TEMPERATURE`: 0.1

## Common Commands

### Setup and Installation

```bash
# Create and activate conda environment
conda create -n ollama_rag python=3.12
conda activate ollama_rag

# Install dependencies (order matters)
conda install pytorch  # First, with CUDA support if available
conda install ollama
pip install -U langchain
conda install -c conda-forge faiss-gpu  # GPU version
pip install streamlit sentence-transformers pypdf langchain_text_splitters langchain_community langchain_huggingface
```

### Running the Application

```bash
# Start Ollama server (if not running)
ollama serve

# Run the Streamlit app
streamlit run app.py
```

### Ollama Model Management

```bash
# Pull the default LLM model (deepseek-r1:latest)
ollama pull deepseek-r1:latest

# List available models
ollama list

# Pull other models as needed
ollama pull llama2
ollama pull mistral
```

## RAG Pipeline Flow

1. **PDF Upload** → `file_processor.get_pdf_text()` extracts text from uploaded PDFs
2. **Text Chunking** → `file_processor.get_text_into_chunks()` splits text using RecursiveCharacterTextSplitter
3. **Vector Store Creation** → `vector_store.update_vector_store()` creates/updates FAISS index with embeddings
4. **Retrieval** → User query retrieves top-k chunks via FAISS retriever
5. **Prompt Construction** → `rag.create_rag_prompt()` builds prompt with retrieved context
6. **LLM Generation** → `llm.query_llm_stream()` streams response from Ollama

## Important Implementation Notes

- **Embedding Model Caching**: The embedding model is cached using `@st.cache_resource` in `embeddings.py` to avoid reloading on every rerun
- **Session State Management**: Multiple session state variables track processed files, vector store, and RAG settings (chunk size, overlap, retriever k)
- **GPU Support**: FAISS vector store uses GPU if CUDA is available; CPU fallback if not
- **File Deduplication**: PDF files are tracked in `st.session_state.processed_files` to avoid reprocessing
- **Streaming Responses**: Ollama responses are streamed token-by-token for better UX
- **Top-k Testing Feature**: Sidebar includes a test toggle to limit retrieved chunks for LLM prompt (useful for debugging)

## Development Notes

- The application uses a wide layout (`st.set_page_config(layout="wide")`)
- All core modules are imported in `app.py` from the `str/` package
- The `str/` directory structure is organized but some subdirectories (`ui/`, `utils/`) are not yet implemented
- The data directory contains sample PDFs used for testing
