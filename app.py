import os
import pinecone
import streamlit as st
from langchain.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma, Pinecone
from langchain.chains import ConversationalRetrievalChain, LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain import PromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

dataset_path = "./dataset.txt"
loader = TextLoader(dataset_path)
comments = loader.load_and_split()

embeddings = OpenAIEmbeddings(model_name="ada")
vectordb = Chroma.from_documents(comments, embedding=embeddings, persist_directory=".")
vectordb.persist()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Assuming that GPT-4 is used for grammar, structure, and fact-checking
# and Claude is used for providing tips and encouraging students to do their own research
grammar_llm = OpenAI(temperature=0.8)
tips_llm = Claude(temperature=0.8)

grammar_qa = ConversationalRetrievalChain.from_llm(grammar_llm, vectordb.as_retriever(), memory=memory)
tips_qa = ConversationalRetrievalChain.from_llm(tips_llm, vectordb.as_retriever(), memory=memory)



st.title('AI Statement Reviewer')

user_input = st.text_area("Enter your personal statement here:")

if st.button('Get feedback'):
    grammar_result = grammar_qa({"question": user_input})
    tips_result = tips_qa({"question": user_input})
    st.write("Grammar and Structure Feedback:")
    st.write(grammar_result["answer"])
    st.write("Tips and Recommendations:")
    st.write(tips_result["answer"])


