import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from typing import List, Union


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

    def add_documents(
        self,
        summaries: List[str],
        originals: List[Union[str, object]],
        file_name: str,
        element_type: str
    ):
        doc_ids = [str(uuid.uuid4()) for _ in summaries]
        summary_docs = [
            Document(
                page_content=summaries[i],
                metadata={
                    self.id_key: doc_ids[i],
                    "file_name": file_name,
                    "element_type": element_type
                },
            )
            for i in range(len(summaries))
        ]

        self.vector_store.add_documents(summary_docs)
        self.store.mset(list(zip(doc_ids, originals)))
    
    def query(self, query: str, k: int = 8):
        return self.retriever.invoke(query)
    
    def get_retriever(self):
        return self.retriever