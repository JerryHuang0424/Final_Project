import streamlit as st
from api_interface import RAGInterface


def main():
    """Main Streamlit application with clean frontend-backend separation"""

    st.set_page_config(layout="wide")
    st.title("Retriever-Argument-Generator (RAG) Demo")

    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content': "Hello! I'm your RAG assistant. Ask me anything about the documents I've read!"}]

    if 'rag_interface' not in st.session_state:
        st.session_state.rag_interface = RAGInterface()

    # Sidebar for system status and controls
    with st.sidebar:
        st.header("System Status")

        # Initialize system button
        if st.button("Initialize System"):
            with st.spinner("Initializing RAG system..."):
                init_result = st.session_state.rag_interface.initialize_system()

                if init_result['success']:
                    st.success("System initialized successfully!")
                    st.info(f"Vector store: {'✓' if init_result['vector_store_loaded'] else '✗'}")
                    st.info(f"LLM connection: {'✓' if init_result['model_available'] else '✗'}")
                else:
                    st.error(f"Initialization failed: {init_result['error']}")

        # System status display
        st.subheader("Current Status")
        status = st.session_state.rag_interface.get_system_status()

        st.write(f"System initialized: {'✓' if status['initialized'] else '✗'}")
        st.write(f"Vector store loaded: {'✓' if status['vector_store_loaded'] else '✗'}")
        st.write(f"LLM available: {'✓' if status['model_available'] else '✗'}")

        # Clear cache button
        if st.button("Clear Cache"):
            st.session_state.rag_interface.clear_cache()
            st.success("Cache cleared!")

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # Chat input
    prompt = st.chat_input('Ask about the documents')

    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process query with RAG interface
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Check if system is initialized
                status = st.session_state.rag_interface.get_system_status()

                if not status['initialized']:
                    st.error("System not initialized. Please click 'Initialize System' in the sidebar first.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "System not initialized. Please click 'Initialize System' in the sidebar first."
                    })
                else:
                    # Process the query
                    result = st.session_state.rag_interface.process_query(prompt)

                    if result['success']:
                        # Display streaming response
                        response_placeholder = st.empty()
                        full_response = ""

                        for chunk in result['llm_response']:
                            full_response += chunk
                            response_placeholder.markdown(full_response)

                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response
                        })
                    else:
                        st.error(f"Error processing query: {result['error']}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {result['error']}"
                        })


if __name__ == "__main__":
    main()
