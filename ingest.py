from src.utils.summarizer import MultiModalSummarizer
from src.utils.vector_db import VectorDB
from src.services.ingestion.ingestion import Ingestion


if __name__ == "__main__":
    db = VectorDB(persist_directory="./chroma_db")
    summarizer = MultiModalSummarizer()
    ingestor = Ingestion(db, summarizer)

    ingestor.ingest_directory("./content")
    db.persist()
    print("Ingestion completed and vector DB persisted.")