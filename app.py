import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
   
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text

def pdf_file(file):
    loader = PyPDFLoader(file)
    data = loader.load_and_split()
    return data

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_documents(text_chunks,embedding = embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template="""
    answer the question as detailed as possible from the provided context, make sure to provide all the
    details. if not provided, context just say "answer is not available in the context", dont provide the wrong answer \n
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro",temperature = 0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context","question"])
    chain = load_qa_chain(model,chain_type = "stuff",prompt = prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question}
        ,return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat Vista")
    st.header("chat with Vista using Gemini")

    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu: ")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if not os.path.exists('temp_folder'):
                    os.makedirs('temp_folder')
                file_path = os.path.join('temp_folder', pdf_docs.name)

                with open(file_path, "wb") as f:
                    f.write(pdf_docs.getbuffer())

                file_path = os.path.abspath(file_path)
                print(file_path)
                raw_text = pdf_file(file_path)

                
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()