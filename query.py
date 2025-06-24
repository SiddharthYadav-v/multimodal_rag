from src.utils.vector_db import VectorDB
from src.services.qa.qa_chain import build_qa_chain

if __name__ == "__main__":
    db = VectorDB(persist_directory="./chroma_db")
    retriever = db.get_retriever()
    qa_chain = build_qa_chain(retriever)

    while True:
        question = input("\nAsk a question (or type 'exit'):")
        if question.lower() == "exit":
            break
        result = qa_chain.invoke(question)
        print("\nAnswer:\n", result)