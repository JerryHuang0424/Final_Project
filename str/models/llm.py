import streamlit as st
import ollama
import constants

def check_ollama_model_availbility():
    # 会先检查session——state中是否有：Ollama_model_checked_this_session，如果有且为True，说明已经检查过了，就直接返回之前的结果。
    # 如果没有检查过，就连接Ollama，列出可用模型，看看我们需要的模型是否在其中，并把结果存到session_state中，以便下次直接使用。最后返回模型是否可用的布尔值。
    if st.session_state.get("Ollama_model_checked_this_session", False):
        return st.session_state.ollama_model_available
    st.sidebar.write(f"Checking availability of Ollama model: `{constants.LLM_MODEL_NAME}`...")
    try:
        client = ollama.Client(host = constants.OLLAMA_HOST )
        list_response = client.list()
        available_model_tags = [model_obj.model for model_obj in list_response.models if hasattr(model_obj, 'model')] if hasattr(list_response, 'models') and isinstance(list_response.models, list) else []
        st.write(f"Available Ollama models: {available_model_tags}")
        if constants.LLM_MODEL_NAME in available_model_tags:
            st.sidebar.success(f"Ollama model `{constants.LLM_MODEL_NAME}` is available!")
            st.session_state.ollama_model_available = True
        else:
            st.sidebar.error(f"Ollama model '{constants.LLM_MODEL_NAME}' not found!")
            st.sidebar.info(f"Available models: {available_model_tags}")
            st.session_state.ollama_model_available = False
    except Exception as e:
        st.sidebar.error(f"Error connecting to Ollama or listing models: {e}")
        st.session_state.ollama_model_available = False
    #之前没有这个变量，现在加上，表示已经检查过了，避免重复检查浪费资源。
    st.session_state.Ollama_model_checked_this_session = True
    return st.session_state.ollama_model_available



def query_llm_stream(prompt, model_name = constants.LLM_MODEL_NAME):
    if not st.session_state.ollama_model_available:
        yield "Ollama model is not available. Please check the sidebar for details."
        return
    try:
        client = ollama.Client(host=constants.OLLAMA_HOST)
        llm_options = {"temperature": st.session_state.get("llm_temperature", constants.LLM_TEMPERATURE)}
        stream = client.chat(model = model_name, messages = [{"role": "user", "content": prompt}], stream = True, options = llm_options)
        for chunks in stream:
            if "message" in chunks and "content" in chunks["message"]:
                yield chunks["message"]["content"]
            if chunks.get('done', False) and chunks.get('error'):
                st.error(f'Error from Ollama during streaming: {chunks["error"]}')
                break
    except ollama.ResponseError as e:
        error_message = f"Ollama API error: {e.status_code} - {str(e)}"
        st.error(error_message)
        yield f"Ollama Error: {str(e)}"
    except Exception as e:
        st.error(f"Unexpected error while querying Ollama: {e}")
        st.exception(e)
        yield f"[Connection or other error: {e}]"
        