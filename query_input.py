from langchain.chains import RetrievalQA
from langchain_chroma import Chroma 
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from load_document import get_embedding_func


CHROMA_PATH = "Chroma"

TEMPLATE = """
    Use the following context to answer questions:
    {context}
    ---
    Question:{question}
    ANSWER:
    """

def get_context(query_text:str):
    # get the relevant context from the vector database
    context_text = ""
    embedding_func = get_embedding_func()

    # load the vector database from the persisted directory
    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_func,
    )

    # get the result based on similarity search (k=1 means we get most similar documents) 
    result = vector_db.similarity_search(query_text, k=1)

    # format the context text from the results
    context_text = "\n\n---\n\n".join([doc.page_content for doc in result])
    return context_text, result

def generate_prompt(query_text:str, context_text:str):
    # create the prompt template
    prompt_template = ChatPromptTemplate.from_template(TEMPLATE)
    prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )
    # print(f"Prompt: {prompt}")
    return prompt


def get_response(prompt:str, result:list):
    # invoke the LLM with the prompt to search for relevant answer response
    model = OllamaLLM(model='llama3.2')
    response_text = model.invoke(prompt)
    
    # get sources from the result (book name)
    sources = [doc.metadata.get('source') for doc in result]

    formatted_response = f"{response_text}"
    return formatted_response, sources

def main():
    question = input("Ask a question: ")
    context_text, result = get_context(question)
    prompt = generate_prompt(question, context_text)
    get_response(prompt, result)
    # print(f"{response}: \n{source[0] if source else 'No sources found.'}")

if __name__ == "__main__":
    main()