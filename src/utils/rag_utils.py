from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode

SUMMARIZE_TEXT_TABLE_PROMPT_TEMPLATE = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additional comment.
Do no start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""

DESCRIBE_IMAGE_PROMPT_TEMPLATE = """
Describe the image in detail. For context, the image can be part of a research paper explaining different scientiic contexts.
Be specific about graphs, and other mathematical elements.
"""


def summarize_text(texts, model_name: str = "llama-3.1-8b-instant"):
    summarize_text_prompt = ChatPromptTemplate.from_template(
        SUMMARIZE_TEXT_TABLE_PROMPT_TEMPLATE
    )

    model = ChatGroq(temperature=0.5, model=model_name)
    summarize_chain = (
        {"element": lambda x: x} | summarize_text_prompt | model | StrOutputParser()
    )

    text_summaries = summarize_chain.batch(texts, {"max_concurrent": 3})

    return text_summaries


def summarize_tables(tables, model_name: str = "llama-3.1-8b-instant"):
    summarize_table_prompt = ChatPromptTemplate.from_template(
        SUMMARIZE_TEXT_TABLE_PROMPT_TEMPLATE
    )

    model = ChatGroq(temperature=0.5, model=model_name)
    summarize_chain = (
        {"element": lambda x: x} | summarize_table_prompt | model | StrOutputParser()
    )
    tables_html = [table.metadata.text_as_html for table in tables]

    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

    return table_summaries


def summarize_images(images, model_name: str = "gpt-4o-mini"):
    messages = [
        (
            "user",
            [
                {"type": "text", "text": DESCRIBE_IMAGE_PROMPT_TEMPLATE},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"}
                }
            ],
        )
    ]

    prompt_ = ChatPromptTemplate.from_messages(messages)

    chain = prompt_ | ChatOpenAI(model=model_name) | StrOutputParser()

    image_summaries = chain.batch(images)

    return image_summaries

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["text"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context which can include text, tables and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )
