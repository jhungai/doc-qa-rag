# src/config.py
"""Centralized configuration for the RAG pipeline."""

# Retrieval
SIMILARITY_THRESHOLD = 0.45
TOP_K = 5

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "claude-sonnet-5"

# ChromaDB
CHROMA_PERSIST_PATH = "./chroma_db"
COLLECTION_NAME = "policy_docs"

# Log file path
LOG_FILE_PATH = "query_log.jsonl"

