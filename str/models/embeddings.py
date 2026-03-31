from langchain_huggingface import HuggingFaceEmbeddings 
import torch
import streamlit as st


@st.cache_resource
def get_embedding_model():
    from str.constants import EMBEDDING_MODEL
    #st.write(f'Attempting to load embedding model:{EMBEDDING_MODEL}')

    model_kwargs_dict = {"device": "cpu"}
    if torch.cuda.is_available():
        model_kwargs_dict["device"] = "cuda"
        #st.write(f"Pytorch CUDA is available. Current device is : {torch.cuda.get_device_name(0)}")
    else:
        st.write("Pytorch CUDA is not available. Using CPU instead.")
    
    #model implement
    try:
        model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs_dict)
        #st.write(f'Embedding model loaded. Target device: {model_kwargs_dict["device"]}')
        return model
    except Exception as e:
        #st.error(f"Error loading embedding model: {e}")
        return None

    
