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
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


def format_result(result):
    # Combine the answer from the result with the formatted list of video links
    answer = result['answer']
    response = f"{answer}"

    return response

def get_relevant_image(result):
    sources = set((doc.metadata['source']) for doc in result['context'])
    for source in sources:
        if source.find('image') != -1:
            return source
        else:
            return "data/image/doc231210997_page2_image1.png"


# Function to generate answer
def generate_answer(message, history):
    from PIL import Image
    text_response = rag_chain_with_source.invoke(message)
    formatted_results = format_result(text_response)
    image_path = get_relevant_image(text_response)
    image = Image.open(image_path)
    return formatted_results, image

# Gradio interface to handle both text and image outputs
def gradio_interface(message, history):
    text_response, image = generate_answer(message, history)
    return [text_response, image]

# Define the Gradio Chat Interface
answer_bot = gr.Interface(
    fn=gradio_interface,
    inputs=["text"],
    outputs=[gr.Textbox(), gr.Image(type="pil")],
    live=False,
    title="RAG ChatBot",
    description="Ask about content in a RAG Survey paper!",
    submit_btn="Ask"
)

if __name__ == "__main__":
    answer_bot.launch()