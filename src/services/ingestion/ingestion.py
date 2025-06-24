import os
from typing import List

from src.utils.document_parser import parse_pdf, separate_chunks, extract_images
from src.utils.summarizer import MultiModalSummarizer
from src.utils.vector_db import VectorDB


class Ingestion:
    def __init__(self, db: VectorDB, summarizer: MultiModalSummarizer):
        self.db = db
        self.summarizer = summarizer

    def ingest_file(self, file_path: str):
        file_name = os.path.basename(file_path)
        chunks = parse_pdf(file_path)
        texts, tables = separate_chunks(chunks)
        images = extract_images(chunks)

        # Summarize
        text_summaries = self.summarizer.summarize_texts(texts)
        table_summaries = self.summarizer.summarize_tables(tables)
        image_summaries = self.summarizer.summarize_images(images)

        # Index
        self.db.add_documents(text_summaries, texts, file_name, "text")
        self.db.add_documents(table_summaries, tables, file_name, "table")
        self.db.add_documents(image_summaries, images, file_name, "image")

    def ingest_directory(self, folder_path: str):
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                self.ingest_file(os.path.join(folder_path, file))
