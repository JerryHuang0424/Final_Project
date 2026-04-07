from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import pathlib
import torch
import os
import glob
from langchain_huggingface import HuggingFaceEmbeddings

def get_all_pdf_files(directory_path):
    """Get all PDF files in the specified directory"""
    pdf_pattern = os.path.join(directory_path, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    return pdf_files

def get_embbeding_model(EMBEDDING_MODEL):
    model_kwargs_dict = {"device": "cpu"}
    if torch.cuda.is_available():
        model_kwargs_dict["device"] = "cuda"
        #st.write(f"Pytorch CUDA is available. Current device is : {torch.cuda.get_device_name(0)}")
    else:
        print("Pytorch CUDA is not available. Using CPU for embedding model.")
    
    #model implement
    try:
        model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs_dict)
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

def load_and_process_pdf(pdf_path_or_directory = r"data\raw_data"):
    """Load and process PDF file(s) - can accept a single file path or a directory path
    After successful processing, PDF files are removed from the raw_data folder."""

    # Determine if input is a single file or directory
    if isinstance(pdf_path_or_directory, (str, pathlib.Path)):
        if os.path.isdir(pdf_path_or_directory):
            # Process all PDFs in directory
            pdf_files = get_all_pdf_files(pdf_path_or_directory)
            print(f"Found {len(pdf_files)} PDF files in directory: {pdf_path_or_directory}")
        else:
            # Single file
            pdf_files = [pdf_path_or_directory]
    elif isinstance(pdf_path_or_directory, list):
        # Already a list of files
        pdf_files = pdf_path_or_directory
    else:
        raise ValueError("Input must be a file path, directory path, or list of file paths")

    if not pdf_files:
        print("No PDF files found to process.")
        return

    all_texts = []
    processed_files = []

    for pdf_file in pdf_files:
        print(f"\nProcessing file: {os.path.basename(pdf_file)}")
        documents = []

        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            print(f"  Loaded {len(documents)} documents successfully.")
        except Exception as e:
            print(f"  Error loading PDF {pdf_file}: {e}")
            continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        if not documents:
            print(f"  No documents to split for {pdf_file}.")
        else:
            texts = text_splitter.split_documents(documents)
            all_texts.extend(texts)
            processed_files.append(pdf_file)
            print(f"  Split into {len(texts)} chunks successfully.")

    if all_texts:
        print(f"\nTotal chunks from all files: {len(all_texts)}")
        embeddings = get_embbeding_model('sentence-transformers/all-mpnet-base-v2')
        Chroma.from_documents(all_texts, embeddings, persist_directory=r"data\chroma", collection_name="Document_vector")
        print("Created Chroma vector store successfully with all documents.")

        # Remove processed files
        if processed_files:
            print("\nRemoving processed PDF files from raw_data folder...")
            for pdf_file in processed_files:
                try:
                    os.remove(pdf_file)
                    print(f"  Removed: {os.path.basename(pdf_file)}")
                except Exception as e:
                    print(f"  Error removing {pdf_file}: {e}")
            print("All processed files removed successfully.")
    else:
        print("No texts to create embeddings from any files.")


def load_data_from_chroma():
    load_and_process_pdf()
    try:
        embeddings = get_embbeding_model('sentence-transformers/all-mpnet-base-v2')
        vector_store = Chroma(persist_directory=r"data\chroma", collection_name="Document_vector", embedding_function=embeddings)
        print("Loaded Chroma vector store successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading Chroma vector store: {e}")
        return None

if __name__ == "__main__":
    # Process all PDF files in the data/raw_data directory
    # raw_data_directory = r"data\raw_data"
    vector_store = load_data_from_chroma()
    print(f'Vector store loaded with {vector_store._collection.count()} documents.')
        
