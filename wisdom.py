#importing the necessary libraries
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import streamlit as st
import openai
from langchain.vectorstores import Pinecone


st.write("Openai api key:", st.secrets["OPENAI_API_KEY"])
st.write("Pinecone api key:", st.secrets["PINECONE_API_KEY"])


#loading the documents from the data directory
directory = "data"

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)
len(documents)


"""splitting the document into smaller chunks to ensure the size of the document is manageable and that no relevant information is missed out"""
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)
print(len(docs))


#displaying the page content of the splitted document
print(docs[10].page_content)

#creating embeddings by converting the splitted chunks of text into a format the AI model can understand
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


#checking the dimension of the embedded query
query_result = embeddings.embed_query("Hello world")
len(query_result)


#storing the embeddings in vector database pinecone
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"], 
    environment="gcp-starter"
)
index_name = "languagetutor-chatbot"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name) 


# pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment="gcp-starter")
# pinecone.list_indexes()


#accessing the embedding by using the similarity search function
def get_similar_docs(query, k=1, score= False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

query = "How are you? in Yoruba."
similar_docs = get_similar_docs(query)
similar_docs
