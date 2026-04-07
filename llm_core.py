from openai import OpenAI
from config import Config
from rag_core import rag_retrieve


class LLMCore:
    """Core LLM functionality with clean interface and connection management"""

    def __init__(self):
        self.client = None
        self.connection_checked = False
        self.connection_available = False

    def check_connection(self) -> bool:
        """
        Check LLM API availability and cache the result

        Returns:
            bool: True if connection is available, False otherwise
        """
        print("Checking Silicon Flow API connection...")

        try:
            client = OpenAI(
                api_key=Config.get_api_key('third_party'),
                base_url=Config.get_api_url('third_party')
            )

            # Send a simple test request to verify connection
            test_response = client.chat.completions.create(
                model=Config.get_default_model(),
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10
            )
            print(f"Silicon Flow API connection successful: {test_response.choices[0].message.content}")

            self.client = client
            self.connection_checked = True
            self.connection_available = True
            return True

        except Exception as e:
            print(f"Error connecting to Silicon Flow API: {e}")
            print("Please check your API key and network connection")
            self.connection_checked = True
            self.connection_available = False
            return False

    def query_stream(self, prompt, model_name=None):
        """
        Query the LLM with streaming response

        Args:
            prompt (str): The prompt to send to the LLM
            model_name (str): Model name to use (defaults to config)

        Returns:
            generator: Streaming response chunks
        """
        if not self.connection_checked:
            # Check connection if not already done
            if not self.check_connection():
                yield "API connection is not available. Please check your API configuration."
                return

        if not self.connection_available:
            yield "API connection is not available. Please check your API configuration."
            return

        try:
            if model_name is None:
                model_name = Config.get_default_model()

            # Create streaming response
            stream = self.client.chat.completions.create(
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

    def is_ready(self):
        """Check if LLM core is ready for queries"""
        return self.connection_available and self.client is not None


# Legacy functions for backward compatibility
def check_llm_availbility():
    """Legacy function - use LLMCore.check_connection() instead"""
    llm_core = LLMCore()
    return llm_core.check_connection()


def query_llm_stream(prompt, model_name=Config.get_default_model()):
    """
    Legacy function that creates a temporary LLMCore instance
    This maintains backward compatibility during migration
    """
    llm_core = LLMCore()

    # Check connection if not already done
    if not llm_core.connection_checked:
        llm_core.check_connection()

    return llm_core.query_stream(prompt, model_name)


def test_rag_function():
    """Test function to demonstrate RAG pipeline"""

    # Test with new class-based approach
    llm_core = LLMCore()

    test_queries = [
        "How do cats use their sense of smell?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print(f"{'='*60}")

        # Get the RAG-retrieved information
        rag_info = rag_retrieve(query)
        print(f"RAG Retrieved Info:\n{rag_info}")

        # Get the generator object for streaming
        print("\nLLM Response:")
        full_response = ""

        try:
            for chunk in llm_core.query_stream(rag_info):
                print(chunk, end="", flush=True)  # Print each chunk as it comes
                full_response += chunk
            print()  # New line after stream ends
        except Exception as e:
            print(f"\nError during streaming: {e}")
            full_response = f"Error: {e}"

        print(f"\n{'='*60}")


if __name__ == "__main__":
    # Test LLM pipeline
    test_rag_function()