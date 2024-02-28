from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

openai.api_key =st.secrets["OPENAI_API_KEY"]
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment='gcp-starter')
index = pinecone.Index('languagetutor-chatbot')

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


#creating embeddings by converting the splitted chunks of text into a format the AI model can understand
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


#storing the embeddings in vector database pinecone
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"], 
    environment="gcp-starter"
)
index_name = "languagetutor-chatbot"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name) 


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


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string