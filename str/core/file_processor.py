import streamlit as st
from pypdf import PdfReader

def get_pdf_text(new_files):
    #st.write("Extracting text from uploaded PDF(s)...")
    text = ""
    new_file_processed_this_run = []
    for pdf in new_files:
        if pdf.name not in st.session_state.processed_files:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                st.session_state.processed_files.add(pdf.name)
                new_file_processed_this_run.append(pdf.name)
            except Exception as e:
                st.sidebar.error(f"Error reading {pdf.name}: {e}")
    st.write("Code is execute into PDF processer.")
    # if new_file_processed_this_run: 
    #     st.write(f"Processed new file(s): {', '.join(new_file_processed_this_run)}")
    return text


def get_text_into_chunks(text, chunk_size_val, chunk_overlap_val):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_val, chunk_overlap=chunk_overlap_val)
    chunks = text_splitter.create_documents([text])
    return chunks