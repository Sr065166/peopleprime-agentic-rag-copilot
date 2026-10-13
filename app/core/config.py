from functools import lru_cache
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "PeoplePrime HR Policy Agentic RAG Copilot"
    app_env: str = "development"
    app_version: str = "1.0.0"

    # LLM Settings (Groq API)
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Vector DB Settings (Pinecone)
    pinecone_api_key: str = ""
    pinecone_index_name: str = "peopleprime-hr-kb-384"
    pinecone_namespace: str = "peopleprime-policies"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Embedding Settings (Local Hugging Face model)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # External Web Search (Tavily Fallback)
    tavily_api_key: str = ""

    # Retrieval & Workflow
    top_k: int = 4
    max_retries: int = 1

    # Security
    admin_api_key: str = "change-me-in-production"

    # Ingestion constraints
    max_upload_size_bytes: int = 15 * 1024 * 1024  # 15 MB
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # File paths
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator(
        "groq_api_key",
        "groq_model",
        "pinecone_api_key",
        "pinecone_index_name",
        "pinecone_namespace",
        "pinecone_cloud",
        "pinecone_region",
        "embedding_model",
        "tavily_api_key",
        "admin_api_key",
        mode="before",
    )
    @classmethod
    def clean_env_strings(cls, v: str | None) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            s = v.strip()
            # Strip matching quotes if present
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()
            return s
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
