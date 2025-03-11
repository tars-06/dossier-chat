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
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say, "answer is not available in the context". Don't provide incorrect information.
    
    Context:
    {context}
    
    Question:
    {question}
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.session_state['last_question'] = user_question
        st.session_state['last_response'] = response["output_text"]
        
        st.markdown(f"**You:** {user_question}")
        st.markdown(f"**Bot:** {response['output_text']}")
    except Exception as e:
        if '429' in str(e):
            st.error("Rate limit exceeded. Please try again later.")
            time.sleep(60)
        else:
            st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config(page_title="Dossier Chat")
    st.title("Dossier Chat")
    
    if 'last_question' not in st.session_state:
        st.session_state['last_question'] = None
    if 'last_response' not in st.session_state:
        st.session_state['last_response'] = None
    
    if st.session_state['last_question'] and st.session_state['last_response']:
        st.markdown(f"**Previous Question:** {st.session_state['last_question']}")
        st.markdown(f"**Previous Answer:** {st.session_state['last_response']}")
    
    user_question = st.text_input("Ask a question from the PDF files")
    if user_question and user_question != st.session_state.get('last_question', ''):
        user_input(user_question)
    
    with st.sidebar:
        st.title("Resources")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
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
