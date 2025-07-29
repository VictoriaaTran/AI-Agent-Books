from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DATA_PATH = "Data"
CHROMA_PATH = "Chroma"

# load PDFs documents
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = document_loader.load()
    return documents

# split the text in the documents into chunks
def split_documents(documents:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    # split the documents into chunks
    return text_splitter.split_documents(documents)

# creating embeddings for the text
def get_embedding_func():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

# call the functions to load and split the documents
documents = load_documents()
chunks = split_documents(documents)

# creating the vector database using Chroma
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=get_embedding_func(),
    persist_directory=CHROMA_PATH,
)

# checking the number of documents in the vector database
# print(vector_db._collection.count())
