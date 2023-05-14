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


# Create a function to get the feedback from the AI model
def get_feedback(statement, format="pdf"):
    # Get the predictions from the AI model
    predictions = model.predict(statement)

    # Create a list of feedback
    feedback = []
    for prediction in predictions:
        feedback.append(prediction["feedback"])

    return feedback

# Create a function to display the feedback
def display_feedback(feedback):
    st.header("📝 Here's your feedback:")
    st.write(feedback)

# Create a main function
def main():
    """
    AI Statement Reviewer App

    This application uses OpenAI's GPT-4 and Anthropic's Claude language models, along with the LangChain framework, to provide an AI-driven statement review service for students applying to universities. 

    The app reviews student personal statements and gives feedback on several aspects:
    1. Grammar and Structure: The app evaluates the grammar and structural integrity of the personal statement, providing suggestions for improvement where necessary.
    2. Tips and Recommendations: The app gives personalized tips and recommendations, encouraging students to engage in their own research and further study.

    The application is designed to democratize access to high-quality personal statement advice, providing feedback that was previously only available to a select few. The ultimate goal is to enhance social mobility and level the playing field for all students, regardless of their background.

    This application is a part of Afinity.io's suite of student advice services, which aim to provide comprehensive guidance to students choosing courses at the university level. It is also an important step towards Afinity.io's vision for 2030: to have every student make informed study choices through the Afinity platform.
    """

    load_dotenv(find_dotenv())

    st.set_page_config(
        page_title="AI Statement Reviewer",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title('🎓 AI Statement Reviewer 📝')

    st.header('By Afinity.io')

    st.markdown("""
    This application uses AI to review and provide feedback on your university personal statement! 

    Here's what it does:

    1. 🧐 **Review your grammar and structure**: The AI, powered by GPT-4, will check your statement for any grammatical or structural issues.
    2. 💡 **Provide tips and recommendations**: Claude, another advanced AI, gives personalized tips and recommendations to make your statement even better.

    This tool is part of Afinity.io's mission to democratize access to high-quality advice for students, no matter their background.

    Just upload your personal statement below, and let our AI give you feedback!
    """)

    uploaded_file = st.file_uploader("Upload your personal statement here", type=["txt", "docx", "pdf"])
    text_input = st.text_area("Or paste your personal statement here:")

    if uploaded_file is not None:
        statement = uploaded_file.read().decode()
        file_type = uploaded_file.type.split('/')[1]
        if st.button('Get feedback for uploaded file'):
            feedback = get_feedback(statement, file_type)
            display_feedback(feedback)
    elif text_input:
        if st.button('Get feedback for pasted text'):
            feedback = get_feedback(text_input, "text")
            display_feedback(feedback)
    else:
        st.write("📤 Please upload your personal statement or paste it into the text box.")
    

if __name__ == "__main__":
    main() 