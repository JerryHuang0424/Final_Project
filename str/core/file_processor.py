import streamlit as st
from pypdf import PdfReader

def get_pdf_text(file):
    #st.write("Extracting text from uploaded PDF(s)...")
    text = ""
    try:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
             text += page.extract_text() + "\n"
    except Exception as e:
        st.sidebar.error(f"Error reading {file.name}: {e}")
    return text


def get_text_into_chunks(text, chunk_size_val, chunk_overlap_val):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_val, chunk_overlap=chunk_overlap_val)
    chunks = text_splitter.create_documents([text])
    print(f"Text split into {len(chunks)} chunks.")
    return chunks