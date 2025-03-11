import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv  
from io import BytesIO
import time


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    try:
        if not pdf_docs:
            raise ValueError("No PDF files uploaded.")
        
        for pdf in pdf_docs:
            # Read the uploaded file content as bytes
            pdf_bytes = pdf.read()
            pdf_stream = BytesIO(pdf_bytes)
            pdf_reader = PdfReader(pdf_stream)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"An error occurred while processing the PDFs: {str(e)}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding= embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all
    the details, if the answer is not in the provided context just say, "answer is not available in the context". Don't provide the wrong answer \n
    Context: \n {context}/\n
    Question: \n {question}\n
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load the vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True
        )

        # Display user input with a human icon and styled background
        st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 20px;">'
                    f'<img src="https://img.icons8.com/ios-filled/50/000000/user.png" style="width: 30px; height: 30px; margin-right: 10px;" />'
                    f'<div style="padding: 10px; background-color: #f0f0f0; border-radius: 10px; width: auto; max-width: 80%;">{user_question}</div>'
                    f'</div>', unsafe_allow_html=True)

        # Display bot response with a robot icon and styled background
        st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 20px;">'
                    f'<img src="https://img.icons8.com/ios-filled/50/000000/robot.png" style="width: 30px; height: 30px; margin-right: 10px;" />'
                    f'<div style="padding: 10px; background-color: #e0e0e0; border-radius: 10px; width: auto; max-width: 80%;">{response["output_text"]}</div>'
                    f'</div>', unsafe_allow_html=True)

        # Store the question and response in session state to avoid re-execution
        st.session_state['last_question'] = user_question
        st.session_state['last_response'] = response["output_text"]

    except Exception as e:
        # Check for specific 429 error
        if '429' in str(e):
            st.error("Rate limit exceeded. Please try again later.")
            time.sleep(60)  # Wait 1 minute before retrying (adjust as necessary)
        else:
            st.error(f"An error occurred while processing the user input: {str(e)}")

def main():
    st.set_page_config("Polylogue")
    st.markdown("""
    <h1 style="color: #4CAF50; text-align: center;">
        Dossier Chat 
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="50" fill="currentColor" class="bi bi-file-pdf" viewBox="0 0 16 16" style="margin-bottom: 30px">
            <path d="M4 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2zm0 1h8a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1"/>
            <path d="M4.603 12.087a.8.8 0 0 1-.438-.42c-.195-.388-.13-.776.08-1.102.198-.307.526-.568.897-.787a7.7 7.7 0 0 1 1.482-.645 20 20 0 0 0 1.062-2.227 7.3 7.3 0 0 1-.43-1.295c-.086-.4-.119-.796-.046-1.136.075-.354.274-.672.65-.823.192-.077.4-.12.602-.077a.7.7 0 0 1 .477.365c.088.164.12.356.127.538.007.187-.012.395-.047.614-.084.51-.27 1.134-.52 1.794a11 11 0 0 0 .98 1.686 5.8 5.8 0 0 1 1.334.05c.364.065.734.195.96.465.12.144.193.32.2.518.007.192-.047.382-.138.563a1.04 1.04 0 0 1-.354.416.86.86 0 0 1-.51.138c-.331-.014-.654-.196-.933-.417a5.7 5.7 0 0 1-.911-.95 11.6 11.6 0 0 0-1.997.406 11.3 11.3 0 0 1-1.021 1.51c-.29.35-.608.655-.926.787a.8.8 0 0 1-.58.029m1.379-1.901q-.25.115-.459.238c-.328.194-.541.383-.647.547-.094.145-.096.25-.04.361q.016.032.026.044l.035-.012c.137-.056.355-.235.635-.572a8 8 0 0 0 .45-.606m1.64-1.33a13 13 0 0 1 1.01-.193 12 12 0 0 1-.51-.858 21 21 0 0 1-.5 1.05zm2.446.45q.226.244.435.41c.24.19.407.253.498.256a.1.1 0 0 0 .07-.015.3.3 0 0 0 .094-.125.44.44 0 0 0 .059-.2.1.1 0 0 0-.026-.063c-.052-.062-.2-.152-.518-.209a4 4 0 0 0-.612-.053zM8.078 5.8a7 7 0 0 0 .2-.828q.046-.282.038-.465a.6.6 0 0 0-.032-.198.5.5 0 0 0-.145.04c-.087.035-.158.106-.196.283-.04.192-.03.469.046.822q.036.167.09.346z"/>
        </svg>
    </h1>
    <br>
    <br>
    """, unsafe_allow_html=True)

    # Initialize session state variables if they don't exist
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = None
    if 'last_response' not in st.session_state:
        st.session_state['last_response'] = None

    # Check if a previous question and response exist
    if st.session_state['last_question'] and st.session_state['last_response']:
        # Display previous question with a human icon and styled background
        st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 20px; margin-top: 20px;">'
                    f'<img src="https://img.icons8.com/ios-filled/50/000000/user.png" style="width: 30px; height: 30px; margin-right: 10px;" />'
                    f'<div style="padding: 10px; background-color: rgba(0, 0, 0, 0.2); border-radius: 10px; width: auto; max-width: 80%; color: white;">{st.session_state["last_question"]}</div>' 
                    f'</div>', unsafe_allow_html=True)

        # Display previous response with a robot icon and styled transparent background
        st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 20px; margin-top: 20px;">'
                    f'<img src="https://img.icons8.com/ios-filled/50/000000/robot.png" style="width: 30px; height: 30px; margin-right: 10px;" />'
                    f'<div style="padding: 10px; background-color: rgba(0, 0, 0, 0.2); border-radius: 10px; width: auto; max-width: 80%; color: white;">{st.session_state["last_response"]}</div>'
                    f'</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])  # Adjust column width as needed

    with col1:
        # Text input for the user question
        user_question = st.text_input("Ask a question from the PDF files", key="question_input")

    with col2:
        # Add a custom HTML button with SVG icon
        submit_button_html = """
        <button id="submit_button" style="padding: 10px; background-color: #4CAF50; border: none; border-radius: 5px; color: white; cursor: pointer; margin-top: 23px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" class="bi bi-upload" viewBox="0 0 16 16">
                <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5"/>
                <path d="M7.646 1.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 2.707V11.5a.5.5 0 0 1-1 0V2.707L5.354 4.854a.5.5 0 1 1-.708-.708z"/>
            </svg>
        </button>
        """
        # Display custom button with SVG icon
        st.markdown(submit_button_html, unsafe_allow_html=True)

        # Adding JavaScript to trigger the button click in Streamlit
        st.markdown("""
        <script>
        const submitButton = document.getElementById('submit_button');
        submitButton.addEventListener('click', function() {
            window.parent.postMessage({func: 'submit_question'}, "*");
        });
        </script>
        """, unsafe_allow_html=True)

    # Modify the condition to check for both the Enter press and the button click
    if (user_question and  user_question != st.session_state.get('last_question', '')):
        user_input(user_question)

    with st.sidebar:
        st.title("Resources")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.error("No text extracted from the PDFs.")


if __name__ == "__main__":
    main()