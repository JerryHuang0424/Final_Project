from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import pathlib
import torch
from langchain_huggingface import HuggingFaceEmbeddings

pdf_path = pathlib.Path(r"data\raw_data\A Cat's Sense of Smell.pdf")

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

def load_and_process_pdf(pdf_path):

    documents = []

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents successfully.")
    except Exception as e:
        print(f"Error loading PDF: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = []
    if not documents:
        print("No documents to split.")
    else:
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks successfully.")

    if texts:
        # embeddings = OpenAIEmbeddings()
        embeddings = get_embbeding_model('sentence-transformers/all-mpnet-base-v2')
        db = Chroma.from_documents(texts, embeddings, persist_directory=r"data\chroma", collection_name="Document_vector")
        print("Created Chroma vector store successfully.")
    else:
        print("No texts to create embeddings.")

        
