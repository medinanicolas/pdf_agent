import os
from pydantic import BaseModel

class Settings(BaseModel):
    APP_BASE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    RAG_UPLOAD_DIR: str = os.path.join(APP_BASE_DIR, "uploads", "rag")

    NEO4J_URI: str = os.getenv('NEO4J_URI', "")
    NEO4J_USERNAME: str = os.getenv('NEO4J_USERNAME', "")
    NEO4J_PASSWORD: str = os.getenv('NEO4J_PASSWORD', "")
    NEO4J_DATABASE: str = os.getenv('NEO4J_DATABASE', "")

    VECTOR_INDEX_NAME: str = 'ChunkEmbedding'
    VECTOR_DOCUMENT_NODE: str = 'Document'
    VECTOR_CHUNK_NODE: str = 'Chubk'
    VECTOR_SOURCE_PROPERTY: str = 'text'
    VECTOR_EMBEDDING_PROPERTY: str = 'textEmbedding'

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDINGS_URI: str = "https://api.openai.com/v1/embeddings"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

settings = Settings()

