import os
import gradio as gr

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import config

OPENAI_APIKEY = config.OPENAI_API_KEY

# Instantiate embeddings model
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_APIKEY, model='text-embedding-3-large', max_retries=100, chunk_size=16, show_progress_bar=False)
# Instantiate chat model
llm = ChatOpenAI(api_key=OPENAI_APIKEY, temperature=0.5, model='gpt-4-turbo-2024-04-09')
# # load chroma from disk
vectorstore = Chroma(persist_directory="rag_db/", embedding_function=embeddings_model)
# Set up the vectorstore to be the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":10})
# Get pre-written rag prompt
prompt = hub.pull("rlm/rag-prompt")
# Format docs function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to generate answer
def generate_answer(message, history):
    from PIL import Image
    result = rag_chain.invoke(message)
    return result


answer_bot = gr.ChatInterface(
                            generate_answer,
                            chatbot=gr.Chatbot(height=600),
                            textbox=gr.Textbox(placeholder="Ask me a question about RAG", container=False, scale=7),
                            title="RAG ChatBot",
                            description="Ask about content in a RAG Survey paper!",
                            theme="soft",
                            cache_examples=False,
                            retry_btn=None,
                            undo_btn=None,
                            clear_btn=None,
                            stop_btn="Interrupt",
                            submit_btn="Ask"
                        )


if __name__ == "__main__":
    answer_bot.launch()