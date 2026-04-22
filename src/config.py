from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Base project root (where .git lives)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # DATA_DIR as BASE_DIR / "data"
    DATA_DIR: Path = Path(BASE_DIR) / "data"

    # INDEX_DIR as DATA_DIR / "index"
    INDEX_DIR: Path = Path(DATA_DIR) / "index"

    # INDEX_FILE as INDEX_DIR / "?faiss"
    INDEX_FILE: Path = Path(INDEX_DIR)/"index.faiss"

    # METADATA_FILE as INDEX_DIR / "???.pkl"
    METADATA_FILE: Path = Path(INDEX_DIR)/"metadata.pkl"

    class Config:
        env_file = ".env"
        
settings = Settings()