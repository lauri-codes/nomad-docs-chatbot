from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from pathlib import Path

parent = Path(__file__).parent.resolve()
persist_directory = parent / 'data/embeddings/'
doc_directory = parent / 'data/docs/'

def create_embeddings(embedding):
    """
    Setup the the index from with information is retrieved as context for the RAG.
    Here we transform the data into GPT4All embeddings and store it in ChromaDB.
    NOTE: These embeddings are used in the retrieval part, and do not affect the
    LLM models itself. Maybe there is room to explore different embedding models?
    """

    paths = [pth for pth in doc_directory.glob('**/*.md')]
    docs = [UnstructuredMarkdownLoader(str(path)).load() for path in paths]
    docs_list = [item for sublist in docs for item in sublist]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0
    )
    doc_splits = splitter.split_documents(docs_list)
    collection_name, embedding_model = get_embedding_model(embedding)
    Chroma.from_documents(
        documents=doc_splits,
        collection_name=collection_name,
        embedding=embedding_model,
        persist_directory=str(persist_directory)
    )

def get_embedding_model(embedding):
    embedding_options = {
        'gpt4all': ('gpt4all', GPT4AllEmbeddings(device='gpu'))
    }
    try:
        embedding_data = embedding_options[embedding]
    except KeyError:
        raise ValueError(f"Invalid embedding option, choose one of the following: {list(embedding_options.keys())}")

    return embedding_data


def get_retriever(embedding, k):
    collection_name, embedding_model = get_embedding_model(embedding)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embedding_model
    ).as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    create_embeddings('gpt4all')
