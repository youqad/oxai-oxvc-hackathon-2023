import os
import streamlit as st
# from langchain.document_loaders import PyPDFLoader, Docx2txtLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings   
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from dotenv import load_dotenv, find_dotenv
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
import PyPDF2
import docx

# Load environment variables
load_dotenv(find_dotenv()) 

st.set_page_config(
        page_title="AI Statement Reviewer",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )

# Models for selection 
models = ['GPT-3.5', 'GPT-4', 'Claude']

# Create models  
chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0.3)
claude = ChatAnthropic()

# Initialize session state
if 'model' not in st.session_state:
    st.session_state['model'] = 'GPT-3.5-turbo'

# Get model based on user selection
selected_model = st.session_state['model']
if selected_model == 'GPT-3.5':
    llm = chatgpt 
elif selected_model == 'GPT-4':
    llm = gpt4
elif selected_model == 'Claude':
    llm = claude

# Add model selection to sidebar
st.sidebar.header('Select Model') 
selected_model = st.sidebar.selectbox('Choose AI model', models)
st.session_state['model'] = selected_model


# Universities and Majors for selection
universities = ["Infer from statement", "University of Aberdeen", "Abertay University", "Aberystwyth University", "Anglia Ruskin University", "Arden University", "Arts University Bournemouth", "Aston University", "Bangor University", "Bath Spa University", "University of Bath", "University of Bedfordshire", "Birkbeck, University of London", "Birmingham City University", "University of Birmingham", "Bishop Grosseteste University", "University of Bolton", "Bournemouth University", "University of Brighton", "University of Bristol", "British Academy of Jewellery", "British College of Osteopathic Medicine", "Brunel University London", "University of Buckingham", "Buckinghamshire New University", "University of Cambridge", "Canterbury Christ Church University", "Cardiff Metropolitan University", "Cardiff University", "University of Chester", "University of Chichester", "City, University of London", "Coventry University", "Cranfield University", "University for the Creative Arts", "University of Cumbria", "De Montfort University", "University of Derby", "University of Dundee", "Durham University", "University of East Anglia", "University of East London", "Edge Hill University", "Edinburgh Napier University", "University of Edinburgh", "University of Essex", "University of Exeter", "Falmouth University", "Glasgow Caledonian University", "Glasgow School of Art", "University of Glasgow", "University of Gloucestershire", "Goldsmiths, University of London", "University of Greenwich", "Harper Adams University", "Heriot-Watt University", "University of Hertfordshire", "University of the Highlands and Islands", "University of Huddersfield", "University of Hull", "Imperial College London", "Keele University", "University of Kent", "King's College London, University of London", "Kingston University", "Lancaster University", "Leeds Arts University", "Leeds Beckett University", "Leeds Trinity University", "University of Leeds", "University of Leicester", "University of Lincoln", "Liverpool Hope University", "Liverpool Institute for Performing Arts", "Liverpool John Moores University", "University of Liverpool", "London Metropolitan University", "London School of Economics and Political Science, University of London", "London School of Hygiene and Tropical Medicine, University of London", "London South Bank University", "Loughborough University", "Manchester Metropolitan University", "University of Manchester", "Middlesex University", "Newcastle University", "Newman University", "University of Northampton", "Northeastern University", "University of Northumbria at Newcastle", "Norwich University of the Arts", "Nottingham Trent University", "University of Nottingham", "The Open University", "Oxford Brookes University", "University of Oxford", "University of Plymouth", "University of Portsmouth", "Queen Margaret University, Edinburgh", "Queen Mary University of London", "Queen's University Belfast", "Ravensbourne University London", "University of Reading", "Regent's University London", "Robert Gordon University", "University of Roehampton", "Rose Bruford College", "Royal Agricultural University", "Royal Central School of Speech and Drama", "Royal College of Art", "Royal College of Music", "Royal Conservatoire of Scotland", "Royal Holloway, University of London", "Royal Veterinary College, University of London", "University of Salford", "University of Sheffield", "Sheffield Hallam University", "University of South Wales", "University of Southampton", "University of St Andrews", "University of London St George's", "University of St Mark and St John", "University of Stirling", "University of Strathclyde", "University of Suffolk", "University of Sunderland", "University of Surrey", "University of Sussex", "Swansea University", "Teesside University", "Trinity Laban Conservatoire of Music and Dance", "Trinity Saint David, University of Wales", "Truro and Penwith College", "St Mary's University, Twickenham", "UCL (University College London)", "Ulster University", "University Campus Oldham", "University of Warwick", "University of Warwickshire", "University of the West of England, Bristol", "University of the West of Scotland", "University of Westminster", "University of Winchester", "Wirral Metropolitan College", "University of Wolverhampton", "University of Worcester", "Writtle University College", "York College", "York St John University", "University of York"]
majors = ["Infer from statement", "Accounting and Finance", "Aeronautical and Manufacturing Engineering", "Agriculture and Forestry", "Anatomy and Physiology", "Anthropology", "Archaeology", "Architecture", "Art and Design", "Biological Sciences", "Building", "Business and Management Studies", "Chemical Engineering", "Chemistry", "Civil Engineering", "Classics and Ancient History", "Communication and Media Studies", "Complementary Medicine", "Computer Science", "Counselling", "Creative Writing", "Criminology", "Dentistry", "Drama Dance and Cinematics", "Economics", "Education", "Electrical and Electronic Engineering", "English", "Fashion", "Film Making", "Food Science", "Forensic Science", "General Engineering", "Geography and Environmental Sciences", "Geology", "Health And Social Care", "History", "History of Art Architecture and Design", "Hospitality Leisure Recreation and Tourism", "Information Technology", "Land and Property Management", "Law", "Linguistics", "Marketing", "Materials Technology", "Mathematics", "Mechanical Engineering", "Medical Technology", "Medicine", "Music", "Nursing", "Occupational Therapy", "Pharmacology and Pharmacy", "Philosophy", "Physics and Astronomy", "Physiotherapy", "Politics", "Psychology", "Robotics", "Social Policy", "Social Work", "Sociology", "Sports Science", "Veterinary Medicine", "Youth Work"]

@st.cache_data
def load_file(files):
    # st.info("`Analysing...`")
    text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text += "".join([page.extract_text() for page in pdf_reader.pages])
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text += stringio.read()
        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            text += " ".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            st.warning('Please provide a text, pdf or docx file.', icon="⚠️")
    return text

# Initialize session state
if 'text' not in st.session_state:
    st.session_state['text'] = ''

template="You are an expert problem statement reviewer with an expertise in getting A-levels students studying {subject} admitted to their dream university: {university}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="Can you give constructive criticism to improve my problem statement: {statement}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=llm, prompt=chat_prompt)

def get_feedback(text, university, major):
    # Use a loading screen 
    with st.spinner('🔄 Generating feedback...'): 
        feedback = chain.predict(subject=major, university=university, statement=text, verbose=True)
    print(feedback)
    return feedback

def display_feedback(feedback):
    st.write("🌟 📝 Here is your AI feedback:")
    # Style the feedback
    st.markdown(f'<p style="font-size: 20px">{feedback}</p>', unsafe_allow_html=True)


def main():
    """
    AI Statement Reviewer App

    This application provides an AI-driven statement review service for students applying to universities. 

    The app reviews student personal statements and gives feedback on several aspects:
    1. Grammar and Structure: The app evaluates the grammar and structural integrity of the personal statement, providing suggestions for improvement where necessary.
    2. Tips and Recommendations: The app gives personalized tips and recommendations, encouraging students to engage in their own research and further study.

    The application is designed to democratize access to high-quality personal statement advice, providing feedback that was previously only available to a select few. The ultimate goal is to enhance social mobility and level the playing field for all students, regardless of their background.

    This application is a part of Afinity.io's suite of student advice services, which aim to provide comprehensive guidance to students choosing courses at the university level. It is also an important step towards Afinity.io's vision for 2030: to have every student make informed study choices through the Afinity platform.
    """
    st.title("🎓 AI Statement Reviewer 📝")
    
    st.header('✨ By Affinity.io ✨')
    
    st.markdown("""
    This application uses AI to review and provide feedback on your university personal statement! 

    Here's what it does:

    1. 🧐 **Review your grammar and structure**: The AI, powered by GPT-4, will check your statement for any grammatical or structural issues.
    2. 💡 **Provide tips and recommendations**: Claude, another advanced AI, gives personalized tips and recommendations to make your statement even better.

    This tool is part of Afinity.io's mission to democratize access to high-quality advice for students, no matter their background.

    Just upload your personal statement below, and let our AI give you feedback!
    """)
    
    # Get file or text input 
    uploaded_file = st.file_uploader("📂 Upload your personal statement here", type=["pdf","docx","txt"], accept_multiple_files=True) 
    text_input = st.text_area("💬 Or enter your personal statement here:", value=st.session_state['text'])
    st.session_state['text'] = text_input
    
    # Get university and major
    chosen_university = st.selectbox('🏛️ Select your University', universities)
    chosen_major = st.selectbox('📘 Select your Major', majors)

    # Get and display feedback
    if uploaded_file is not None: 
        # Load text from file
        text = load_file(uploaded_file)  
        if st.button("🔍 Get Feedback"):
            feedback = get_feedback(text, chosen_university, chosen_major) 
            display_feedback(feedback)
    elif text_input:
        if st.button("🔍 Get Feedback"):
            feedback = get_feedback(text_input, chosen_university, chosen_major) 
            display_feedback(feedback)
    else:
        st.write("📤 Please upload a file or enter your personal statement to get feedback. 📝") 

if __name__ == "__main__": 
    main()
