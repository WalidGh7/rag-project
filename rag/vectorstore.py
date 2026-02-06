from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str  # <-- required
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@postgres:5432/ragdb"
    CHROMA_DIR: str = "./chroma_data"
    CHROMA_COLLECTION: str = "thesis"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

settings = Settings()

# Converting text into vectors
def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,  # <-- explicit
    )

def get_vectorstore(embeddings=None) -> Chroma:
    if embeddings is None:
        embeddings = get_embeddings()

    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        persist_directory=settings.CHROMA_DIR,
        embedding_function=embeddings,
    )


