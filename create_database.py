from load_document import embedding_function
from langchain_chroma import Chroma
from langchain.schema.document import Document


CHROMA_PATH = "Chroma"
def get_chunk_id(chunks):
    """
    create Ids from page number and chunk index (Data/Book_name:0:0)
    result will look like Page Source: Page Number: Chunk Index
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # if current id is same as the last, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        # calculate chunk id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id #set the last chunk as current for next iter

        chunk.metadata['id'] = chunk_id

    return chunks

embeddings = embedding_function
# create database using the embedding function and ChromaDB
def create_database(chunks:list[Document]):
    data = Chroma(
        collection_name="Books",
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # add or update the documnets 
    existing_items = data.get(include=[]) #IDs are included by default
    existing_ids = set(existing_items["ids"]) #no dups

    # get page id
    chunk_ids = get_chunk_id(chunks)

    # add documents that dont already exist in databse
    new_chunks = []
    for chunk in chunk_ids:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)

    new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]

    if new_chunks:
        data.add_documents(new_chunks, ids=new_chunk_ids)
        data.persist()