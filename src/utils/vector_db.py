import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever


class VectorDB:
    def __init__(
        self, id_key: str = "doc_id", collection_name: str = "multi_modal_rag"
    ):
        self.id_key = id_key
        self.store = InMemoryStore()
        self.vector_store = Chroma(
            collection_name=collection_name, embedding_function=OpenAIEmbeddings()
        )
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=self.store,
            id_key=self.id_key,
        )
