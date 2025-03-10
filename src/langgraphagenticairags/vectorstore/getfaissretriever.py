from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def faissretriever(doc_splits):
    vectorstore=FAISS.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings()
    )
    retriever=vectorstore.as_retriever()
    return retriever


