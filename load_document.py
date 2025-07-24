from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings


DATA_PATH = "Data"

# load PDFs documents
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
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
    return text_splitter.split_documents(documents)

# creating embeddings for the text
def get_embedding_func():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


documents = load_documents()
chunks = split_documents(documents)
embedding_function = get_embedding_func()
