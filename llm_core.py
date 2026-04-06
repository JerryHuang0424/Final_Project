from openai import OpenAI
from config import Config
from rag_core import rag_retrieve

def check_llm_availbility():
    # 检查Silicon Flow API连接是否正常

    is_connection_successful = False

    print("Checking Silicon Flow API connection...")
    try:
        client = OpenAI(
            api_key=Config.get_api_key('third_party'),
            base_url=Config.get_api_url('third_party')

            # api_key='sk-uyjxppcyvkvfaqyinuvhaelyrsycerybudbknissurqgkbhx',
            # base_url='https://api.siliconflow.cn/v1'
        )
        # 发送一个简单的测试请求来验证连接
        test_response = client.chat.completions.create(
            model=Config.get_default_model(),
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=10
        )
        print(f"Silicon Flow API connection successful:{test_response.choices[0].message.content}")
        is_connection_successful = True
    except Exception as e:
        print(f"Error connecting to Silicon Flow API: {e}")
        print("Please check your API key and network connection")
        is_connection_successful = False

    return is_connection_successful


def query_llm_stream(prompt, model_name=Config.get_default_model()):
    if not check_llm_availbility():
        yield "API connection is not available. Please check the sidebar for details."
        return

    try:
        client = OpenAI(
            api_key=Config.get_api_key('third_party'),
            base_url=Config.get_api_url('third_party')
        )

        # 创建流式响应
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=Config.get_temperature(),
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
        yield f"API Error: {str(e)}"


def test_rag_function():
    """Test function to demonstrate RAG retrieval"""

    test_queries = [
        "How do cats use their sense of smell?"
        # "What are mutualistic fungi?",
        # "Tell me about transportation transformations",
        # "Why did people abandon hunting and gathering?",
        # "What makes Mercury so dense?",
        # "How do marine mammals adapt to their environment?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print(f"{'='*60}")

        # Get the RAG-retrieved information
        rag_info = rag_retrieve(query)
        print(f"RAG Retrieved Info:\n{rag_info}")

        # Get the generator object for streaming
        stream_generator = query_llm_stream(rag_info)

        # Consume the stream and print the response
        print("\nLLM Response:")
        full_response = ""

        try:
            for chunk in stream_generator:
                print(chunk, end="", flush=True)  # Print each chunk as it comes
                full_response += chunk
            print()  # New line after stream ends
        except Exception as e:
            print(f"\nError during streaming: {e}")
            full_response = f"Error: {e}"

        print(f"\n{'='*60}")

if __name__ == "__main__":
    #Test rag pipeline
    test_rag_function()