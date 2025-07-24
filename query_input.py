from langchain.chains import RetrievalQA
from langchain_chroma import Chroma 
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from load_document import embedding_function


CHROMA_PATH = "Chroma"

TEMPLATE = """
    Use the following context to answer questions:

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

def query_rag(query_text:str):

    # get the db
    embedding_func = embedding_function
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_func,
    )

    # search db for results
    result = db.similarity_search_with_score(query_text, k=1)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in result])
    prompt_template = ChatPromptTemplate.from_template(TEMPLATE)
    prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )

    model = OllamaLLM(model='llama3.2')
    response_text = model.invoke(prompt)

    # sources = [doc.metadata.get('source', None) for doc, _score in result]
    # print('\n')
    formatted_response = f"{response_text}"
    # print(formatted_response)
    return formatted_response

# def main():
#     query_text = input("Question:")
#     print(query_rag(query_text))
            
# main()