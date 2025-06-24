from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode


def parse_docs(docs):
    b64, text = [], []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    question = kwargs["question"]
    context = "".join([t.text for t in docs_by_type["texts"]])

    prompt_content = [
        {
            "type": "text",
            "text": f"Answer based only on the context below:\n{context}\nQuestion: {question}",
        }
    ]

    for image in docs_by_type["images"]:
        prompt_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
        )

    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])


def build_qa_chain(retriever, model_name="gpt-4o-mini"):
    return (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model=model_name)
        | StrOutputParser()
    )
