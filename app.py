import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings   
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from io import StringIO
from langchain.vectorstores import FAISS
import PyPDF2

# Load environment variables
load_dotenv(find_dotenv())  

# Set page config  
st.set_page_config(page_title="AI Statement Reviewer", page_icon="📚")  

@st.cache_data
def load_file(files):
    st.info("`Analysing...`")
    text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text += "".join([page.extractText() for page in pdf_reader.pages])
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text += stringio.read()
        else:
            st.warning('Please provide a text or pdf file.', icon="⚠️")
    return text

# Initialize session state
if 'text' not in st.session_state:
    st.session_state['text'] = ''

# Create models  
claude = ChatAnthropic()

# Create retrieval chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

template="You are an expert problem statement reviewer with an expertise in getting A-levels students studying {subject} admitted to their dream university: {university}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="Can you give constructive criticism to improve my problem statement: {statement}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=llm, prompt=chat_prompt)

def get_feedback(text):
    # Use a loading screen 
    with st.spinner('🔄 Generating feedback...'): 
        feedback = chain.predict(subject="English", university="Oxford University", statement=text, verbose=True)
    print(feedback)
    return feedback

def display_feedback(feedback):
    st.write("🌟 Here is the AI feedback:")
    # Style the feedback
    st.markdown(f'<p style="font-size: 20px">{feedback}</p>', unsafe_allow_html=True)


def main():
    # Set page title and icon
    st.title("🎓 AI Statement Reviewer 📝")
    
    # Add description 
    st.header('✨ By Affinity.io ✨')
    st.markdown("""
        This application uses advanced AI to review and provide feedback on your university personal statement! 👨‍🎓👩‍🎓
        
        Here's what it does:

        1. 🧐 **Review your grammar and structure**: Our AI, powered by OpenAI's Davinci and Anthropic's Claude, will check your statement
        for grammar and structure, helping you present your ideas clearly and effectively.
        2. 💡 **Provide useful tips and recommendations**: The AI will provide insightful tips to strengthen your statement.
        3. 🌐 **Support for all students**: Our goal is to provide high-quality, personalized feedback to students from all backgrounds.  
        
        Just upload your statement or paste it in the box below, and let's get started! 🚀
     """)
    
    # Get file or text input 
    uploaded_file = st.file_uploader("📂 Upload your personal statement here", type=["pdf","docx","txt"], accept_multiple_files=True) 
    text_input = st.text_area("💬 Or enter your personal statement here:", value=st.session_state['text'])
    st.session_state['text'] = text_input
    
    # Get and display feedback
    if uploaded_file is not None: 
        # Load text from file
        text = load_file(uploaded_file)  
        if st.button("🔍 Get Feedback"):
            feedback = get_feedback(text) 
            display_feedback(feedback)
    elif text_input:
        if st.button("🔍 Get Feedback"):
            feedback = get_feedback(text_input) 
            display_feedback(feedback)
    else:
        st.write("Please upload a file or enter your personal statement to get feedback. 📝") 
if __name__ == "__main__": 
    main()
