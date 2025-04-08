"""
This module contains the logic for embedding and vectorizing documents.
Vector search is a method of information retrieval where documents and
 queries are represented as vectors instead of plain text, the database
 is going to be hosted locally using ChromaDB and the vectorization is done using the
 Ollama LLM. In order to quckly look up relevant information, to pass to our model.
"""
""" embedding model """
from langchain_ollama import OllamaEmbeddings
""" vector database """
from langchain_chroma import Chroma
""" pass documents to chromadb """
from langchain_core.documents import Document
import os, pandas as pd

""" load in data """

df = pd.read_csv("data.csv")

""" define the embedding model """
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

""" Check if vectordb already exists """
""" if it does, we don't need to add documents again """
db_location = "./chromedb"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(page_content=row["Title"] + "" + row["Review"], metadata={"rating": row["Rating"], "date": row["Date"]}, id=str(i))
        ids.append(str(i))
        documents.append(document)


""" init vector store """

vector_store = Chroma(
    collection_name="data",
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

""" make vector store persistent """

retrieval = vector_store.as_retriever(
    search_kwargs={
        "k": 10                  # number of documents to return
    },
)