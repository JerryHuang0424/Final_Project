import streamlit as st
import constants
from str.models.embeddings import get_embedding_model
from str.models.llm import check_ollama_model_availbility
from str.core.file_processor import get_pdf_text, get_text_into_chunks
from str.core.vector_store import update_vector_store
from str.models.llm import query_llm_stream
from str.core.rag import create_rag_prompt



#Caching


# set_page_config的作用是定义页面的宽度，默认为”centered“，设置为”wide“可以让页面占满整个屏幕宽度。
st.set_page_config(layout="wide")

st.title("PDF RAG Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload a PDF to start the conversation."}]
if "ollama_model_available" not in st.session_state:
    st.session_state.ollama_model_available = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

check_ollama_model_availbility()


with st.sidebar:
    st.header("Settings")

    st.divider()

    st.header("File Upload 📤")
    embedding_model_instance = get_embedding_model()
    uploaded_file = st.file_uploader("Upload PDF",type="pdf", accept_multiple_files= True)
    file_submit = st.button("Process Uploaded Files", key="process_button")
    if uploaded_file:
        if embedding_model_instance:
            st.success("Embedding model loaded successfully.")
            if file_submit:
                if st.session_state.chunk_size != constants.CHUNK_SIZE or st.session_state.chunk_overlap != constants.CHUNK_OVERLAP:
                    st.sidebar.caption("Note: Chunk size/overlap for new docs is based on current RAG settings.")
                raw_text_from_new_file = get_pdf_text(uploaded_file) 
                # st.session_state.messages.append({"role": "user", "content": f"Uploaded {len(uploaded_file)} file(s). Extracting text and creating chunks..."})
                # st.session_state.messages.append({"role": "assistant", "content": raw_text_from_new_file})
                if raw_text_from_new_file:
                    text_chunks = get_text_into_chunks(raw_text_from_new_file, st.session_state.chunk_size, st.session_state.chunk_overlap)
                    st.write(f"Extracted {len(text_chunks)} chunks from the new file(s).")
                    #st.write(f"First chunk preview:\n{chunks[0].page_content[:500]}...")
                    # Here you would typically create/update your vector store with the new chunks and embeddings
                    if text_chunks:
                        update_vector_store(text_chunks,embedding_model_instance)
                else:
                    st.warning("No new text extracted from the uploaded file(s).")
    st.divider()
    
    if st.button("Clear chat", key="clear_chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. Please upload a new PDF to start a new conversation."}]
        #rerun()函数的作用是重现启动一边app.py文档，但是state_session里面的内容不会发生改变
        st.rerun()

   

    st.divider()

    st.subheader("RAG Chat")
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = constants.CHUNK_SIZE
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = constants.CHUNK_OVERLAP
    if "retriever_k" not in st.session_state:
        st.session_state.retriever_k = constants.RETRIEVER_K

    st.session_state.chunk_size = st.number_input("Chunk Size (chars)", key="sb_chunk_size",  min_value=100, max_value=5000, value=st.session_state.chunk_size, step=100)
    st.session_state.chunk_overlap = st.number_input("Chunk Overlap (chars)", key="sb_chunk_overlap", min_value=0, max_value=1000, value=st.session_state.chunk_overlap, step=50)
    st.session_state.retriever_k = st.number_input("Chunks to Retrieve (k_retriever)", key="sb_retriever_k", min_value=1, max_value=20, value=st.session_state.retriever_k, step=1)
        
    # LLM Settings
    st.subheader("LLM Settings")
    if 'llm_temperature' not in st.session_state: st.session_state.llm_temperature = constants.LLM_TEMPERATURE
    if 'use_top_k_for_llm' not in st.session_state: st.session_state.use_top_k_for_llm = 1 
    if 'enable_llm_top_k_test' not in st.session_state: st.session_state.enable_llm_top_k_test = True 

    st.session_state.llm_temperature = st.slider("LLM Temperature", min_value=0.0, max_value=2.0, value=st.session_state.llm_temperature, step=0.1)
    st.session_state.enable_llm_top_k_test = st.checkbox("TEST: Use only N retrieved chunk(s) for LLM Prompt", value=st.session_state.enable_llm_top_k_test, key="cb_enable_llm_top_k")
    if st.session_state.enable_llm_top_k_test:
        st.session_state.use_top_k_for_llm = st.number_input("N (chunks for LLM prompt if test enabled)", min_value=1, max_value=st.session_state.retriever_k, value=st.session_state.use_top_k_for_llm, step=1, key="ni_use_top_k_for_llm")
    
    if st.session_state.vector_store:
        st.sidebar.success("Knowledge base is ready!")
        st.sidebar.write(f"Total unique files processed: {len(st.session_state.processed_files)}")
    elif uploaded_file and not st.session_state.vector_store : # If tried to process but failed
        st.sidebar.warning("Knowledge base processing may have encountered issues or is empty.")
    else: # Default state
        st.sidebar.info("Upload PDF files to build the knowledge base.")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about the document")

if prompt:
    #把用户输入的内容存储到session_state.messages中，role是"user"，content是用户输入的内容。
    st.session_state.messages.append({"role":"user","content":prompt})
    #在聊天窗口上显示用户输入的内容，role是"user"，content是用户输入的内容。
    with st.chat_message("user"):
        st.write(prompt)

    
    with st.chat_message("assistant"):
        full_response_content = " "
        if not st.session_state.ollama_model_available:
            full_response_content = "Ollama model is not available. Please check the sidebar for details."
            st.write(full_response_content)
        elif st.session_state.vector_store is not None:
            if embedding_model_instance is None:
                full_response_content = "Embedding model is not available. Please check the sidebar for details."
                st.write(full_response_content)
            else:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": st.session_state.retriever_k})
                try:
                    all_relevant_docs = retriever.invoke(prompt) 
                    final_context_chunks_for_llm = []
                    if all_relevant_docs:
                        if st.session_state.enable_llm_top_k_test:
                            num_chunks_to_use = min(st.session_state.use_top_k_for_llm, len(all_relevant_docs))
                            final_context_chunks_for_llm = [doc.page_content for doc in all_relevant_docs[:num_chunks_to_use]]
                            st.info(f"🧪 Using Top-{num_chunks_to_use} retrieved chunk(s) for LLM prompt.")
                        else:
                            final_context_chunks_for_llm = [doc.page_content for doc in all_relevant_docs]
                    
                    with st.expander("Retrieved Chunks (for debugging)"):
                        if all_relevant_docs:
                            st.info(f"Retrieved {len(all_relevant_docs)} relevant chunk(s) from the knowledge base.")
                            for i, doc in enumerate(all_relevant_docs):
                                st.text_area(f"Retrieved Chunk {i+1}", value=doc.page_content, height=150)
                        else:
                            st.write("No relevant chunks retrieved.")
                    
                    if final_context_chunks_for_llm:
                        prompt_with_context = create_rag_prompt(prompt, final_context_chunks_for_llm)
                        placeholder = st.empty()
                        for token in query_llm_stream(prompt_with_context):
                            full_response_content += token
                            placeholder.write(full_response_content)
                        placeholder.write(full_response_content)
                    else:
                        full_response_content = "I don't have enough information in the provided documents to answer that."
                        st.write(full_response_content)

                except Exception as e:
                    st.error(f"Error during RAG pipeline: {str(e)}")
                    st.exception(e)
                    full_response_content = f"An error occurred during RAG processing: {str(e)}"
        else:
            full_response_content = "The knowledge base is empty. Please upload and process PDF files first."
            st.write(full_response_content)


    if full_response_content:
        st.session_state.messages.append({"role":"assistant","content":full_response_content})

