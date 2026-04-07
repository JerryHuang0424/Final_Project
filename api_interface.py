"""
API Interface Layer for Frontend-Backend Communication

This module provides a clean interface between the Streamlit frontend
and the RAG/LLM backend modules, ensuring separation of concerns.
"""

from rag_core import RAGCore
from llm_core import LLMCore
from backend_database import load_data_from_chroma
from config import Config


class RAGInterface:
    """Interface for frontend-backend communication"""

    def __init__(self):
        self.rag_core = RAGCore()
        self.llm_core = LLMCore()
        self.vector_store = None
        self.model_available = False
        self.initialized = False

    def initialize_system(self) -> dict:
        """
        Initialize backend components and return status

        Returns:
            dict: System status with success/error information
        """
        try:
            # Initialize vector store
            print("Initializing vector store...")
            self.vector_store = load_data_from_chroma()

            if self.vector_store is None:
                return {
                    'success': False,
                    'error': 'Could not load Chroma vector store. Please check if the database is properly set up.'
                }

            # Initialize RAG core with vector store
            self.rag_core.set_vector_store(self.vector_store)

            # Check LLM connection
            print("Checking LLM connection...")
            self.model_available = self.llm_core.check_connection()

            self.initialized = True

            return {
                'success': True,
                'vector_store_loaded': self.vector_store is not None,
                'model_available': self.model_available,
                'message': 'System initialized successfully'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'System initialization failed: {str(e)}'
            }

    def process_query(self, user_prompt: str) -> dict:
        """
        Process user query through RAG pipeline

        Args:
            user_prompt (str): The user's question or prompt

        Returns:
            dict: Response with RAG results and LLM response
        """
        if not self.initialized:
            return {
                'success': False,
                'error': 'System not initialized. Call initialize_system() first.'
            }

        try:
            # Step 1: Retrieve relevant documents using RAG
            print(f"Processing query: '{user_prompt}'")
            rag_result = self.rag_core.retrieve_documents(user_prompt)

            if not rag_result['success']:
                return rag_result

            # Step 2: Generate response using LLM
            if not self.model_available:
                return {
                    'success': False,
                    'error': 'LLM connection not available. Please check API configuration.'
                }

            llm_response = self.llm_core.query_stream(rag_result['formatted_text'])

            return {
                'success': True,
                'user_prompt': user_prompt,
                'rag_info': rag_result,
                'llm_response': llm_response,
                'message': 'Query processed successfully'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Query processing failed: {str(e)}'
            }

    def get_system_status(self) -> dict:
        """
        Return current system status

        Returns:
            dict: Current system status information
        """
        return {
            'initialized': self.initialized,
            'vector_store_loaded': self.vector_store is not None,
            'model_available': self.model_available,
            'rag_core_ready': self.rag_core.is_ready() if hasattr(self.rag_core, 'is_ready') else False,
            'llm_core_ready': self.llm_core.is_ready() if hasattr(self.llm_core, 'is_ready') else False
        }

    def clear_cache(self):
        """Clear cached vector store and reset state"""
        if self.rag_core:
            self.rag_core.clear_cache()
        self.vector_store = None
        self.initialized = False
        print("Cache cleared successfully")


def test_interface():
    """Test function for the interface layer"""
    interface = RAGInterface()

    # Test initialization
    print("Testing system initialization...")
    init_result = interface.initialize_system()
    print(f"Initialization result: {init_result}")

    if init_result['success']:
        # Test query processing
        test_queries = [
            "How do cats use their sense of smell?",
            "What are mutualistic fungi?",
            "Tell me about transportation transformations"
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing query: {query}")
            print(f"{'='*60}")

            result = interface.process_query(query)
            print(f"Query result: {result['success']}")

            if result['success']:
                print("LLM Response:")
                for chunk in result['llm_response']:
                    print(chunk, end="", flush=True)
                print()
            else:
                print(f"Error: {result['error']}")

    # Test status
    print("\nSystem status:")
    print(interface.get_system_status())


if __name__ == "__main__":
    test_interface()