import streamlit as st
import constants
from openai import OpenAI

def check_ollama_model_availbility():
    # 检查Silicon Flow API连接是否正常
    if st.session_state.get("api_checked_this_session", False):
        return st.session_state.ollama_model_available

    st.sidebar.write("Checking Silicon Flow API connection...")
    try:
        client = OpenAI(
            api_key=constants.SILICON_FLOW_API_KEY,
            base_url=constants.SILICON_FLOW_BASE_URL
        )
        # 发送一个简单的测试请求来验证连接
        test_response = client.chat.completions.create(
            model=constants.SILICON_FLOW_MODEL,
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=10
        )
        st.sidebar.success("Silicon Flow API connection successful!")
        st.session_state.ollama_model_available = True
    except Exception as e:
        st.sidebar.error(f"Error connecting to Silicon Flow API: {e}")
        st.sidebar.info("Please check your API key and network connection")
        st.session_state.ollama_model_available = False

    st.session_state.api_checked_this_session = True
    return st.session_state.ollama_model_available


def query_llm_stream(prompt, model_name=constants.SILICON_FLOW_MODEL):
    if not st.session_state.ollama_model_available:
        yield "API connection is not available. Please check the sidebar for details."
        return

    try:
        client = OpenAI(
            api_key=constants.SILICON_FLOW_API_KEY,
            base_url=constants.SILICON_FLOW_BASE_URL
        )

        # 创建流式响应
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=st.session_state.get("llm_temperature", constants.LLM_TEMPERATURE),
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

    except Exception as e:
        error_message = f"Silicon Flow API error: {str(e)}"
        st.error(error_message)
        yield f"API Error: {str(e)}"
        