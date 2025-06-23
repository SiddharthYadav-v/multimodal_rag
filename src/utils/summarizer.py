from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


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


class MultiModalSummarizer:
    def __init__(self, text_model_name: str = "llama-3.1-8b-instant", image_model_name: str = "gpt-4o-mini"):
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.text_prompt = ChatPromptTemplate.from_template(SUMMARIZE_TEXT_TABLE_PROMPT_TEMPLATE)
        self.output_parser = StrOutputParser()

    def summarize_texts(self, texts):
        model = ChatGroq(temperature=0.5, model=self.text_model_name)
        summarize_chain = (
            {"element": lambda x: x} | self.text_prompt | model | self.output_parser
        )

        return summarize_chain.batch(texts, {"max_concurrent": 3})
    
    def summarize_tables(self, tables):
        model = ChatGroq(temperature=0.5, model=self.text_model_name)
        summarize_chain = (
            {"element": lambda x: x} | self.text_prompt | model | self.output_parser
        )
        tables_html = [table.metadata.text_as_html for table in tables]
        return summarize_chain.batch(tables_html, {"max_concurrent": 3})
    
    def summarize_images(self, images_b64):
        prompt_messages = [
            (
                "user",
                [
                    {"type": "text", "text": DESCRIBE_IMAGE_PROMPT_TEMPLATE},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
        prompt_ = ChatPromptTemplate.from_messages(prompt_messages)
        chain_ = prompt_ | ChatOpenAI(model=self.image_model_name) | self.output_parser
        return chain_.batch(images_b64)