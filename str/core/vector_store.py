import streamlit as st
import faiss
from langchain_community.vectorstores import FAISS

def update_vector_store(chunks, embedding_model_instance):
    # This function should implement the logic to create/update your vector store with the new chunks and their embeddings
    # For example, you might compute embeddings for each chunk and then add them to a FAISS index or similar structure
    st.write("Updating vector store with new chunks...")
    # Example pseudo-code:
    # for chunk in chunks:
    #     embedding = embedding_model_instance.embed(chunk.page_content)
    #     vector_store.add(embedding, metadata={"text": chunk.page_content})

    if not chunks:
        st.warning("No chunks to add to the vector store.")
        return st.session_state.vector_store
    if  embedding_model_instance is None:
        st.error("Embedding model is not available. Cannot update vector store.")
        return st.session_state.vector_store
    
    try:
        with st.spinner(f"Embedding {len(chunks)} chunks and updating vector store..."):
            if st.session_state.vector_store is None:
                st.session_state.vector_store = FAISS.from_documents(chunks,embedding_model_instance)
                res = faiss.StandardGpuResources()  # Use a single GPU
                st.session_state.vector_store.index = faiss.index_cpu_to_gpu(res, 0, st.session_state.vector_store.index)
                st.write("New vector store created.")
                st.session_state.messages.append({"role": "assistant", "content": st.session_state.vector_store})
            else:
                st.session_state.vector_store.add_documents(chunks,embedding_model_instance)
                st.write("New data add to vector store.")
                st.session_state.messages.append({"role": "assistant", "content": st.session_state.vector_store})

    except Exception as e:
        st.error(f"Error updating vector store: {e}")
    return st.session_state.vector_store