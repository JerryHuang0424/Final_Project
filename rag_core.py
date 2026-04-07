#!/usr/bin/env python3
"""
Core RAG class for retrieving relevant information from Chroma vector store
"""

from backend_database import load_data_from_chroma


class RAGCore:
    """Core RAG functionality with clean interface and state management"""

    def __init__(self):
        self.vector_store = None
        self.initialized = False

    def set_vector_store(self, vector_store):
        """Set the vector store to use for retrieval"""
        self.vector_store = vector_store
        self.initialized = vector_store is not None

    def retrieve_documents(self, user_prompt, top_k=5) -> dict:
        """
        RAG function that retrieves top-k similar chunks from Chroma database
        and formats them for direct LLM input.

        Args:
            user_prompt (str): The user's question or prompt
            top_k (int): Number of top similar chunks to retrieve (default: 5)

        Returns:
            dict: Result with success status and formatted text
        """

        if not self.initialized or self.vector_store is None:
            return {
                'success': False,
                'error': 'Vector store not initialized. Call set_vector_store() first.'
            }

        # Perform similarity search
        try:
            print(f"Searching for similar documents for query: '{user_prompt}'")
            results = self.vector_store.similarity_search(user_prompt, k=top_k)

            if not results:
                return {
                    'success': True,
                    'formatted_text': "No relevant information found in the database for your query.",
                    'documents_found': 0
                }

            print(f"Found {len(results)} relevant document(s)")

            # Format the results for LLM input
            formatted_text = self._format_results_for_llm(results, user_prompt)

            return {
                'success': True,
                'formatted_text': formatted_text,
                'documents_found': len(results),
                'user_prompt': user_prompt
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error during similarity search: {e}"
            }

    def _format_results_for_llm(self, results, user_prompt):
        """
        Format the retrieved chunks into a structured string for LLM input.

        Args:
            results: List of Document objects from similarity search
            user_prompt (str): Original user query

        Returns:
            str: Formatted string ready for LLM input
        """

        # Start with the base instruction
        formatted_text = f"According to user's problem: '{user_prompt}', find following information:\n\n"

        # Add each retrieved chunk with source information
        for i, doc in enumerate(results, 1):
            # Extract source file name from metadata
            source_file = doc.metadata.get('source', 'Unknown Source')
            if '\\' in source_file:
                source_file = source_file.split('\\')[-1]  # Get just the filename

            # Extract page number if available
            page = doc.metadata.get('page', 'Unknown Page')

            formatted_text += f"--- Information {i} (Source: {source_file}, Page: {page}) ---\n"
            formatted_text += f"{doc.page_content}\n\n"

        # Add the final instruction
        formatted_text += "You can only answer based on above information."

        return formatted_text

    def clear_cache(self):
        """Clear the cached vector store"""
        self.vector_store = None
        self.initialized = False
        print("Vector store cache cleared.")

    def is_ready(self):
        """Check if RAG core is ready for queries"""
        return self.initialized and self.vector_store is not None


# Legacy function for backward compatibility
def rag_retrieve(user_prompt, top_k=5):
    """
    Legacy function that creates a temporary RAGCore instance
    This maintains backward compatibility during migration
    """
    rag_core = RAGCore()

    # Load vector store if not already done
    if rag_core.vector_store is None:
        vector_store = load_data_from_chroma()
        if vector_store is None:
            return "Error: Could not load Chroma vector store. Please check if the database is properly set up."
        rag_core.set_vector_store(vector_store)

    result = rag_core.retrieve_documents(user_prompt, top_k)

    if result['success']:
        return result['formatted_text']
    else:
        return result['error']


def clear_vector_store_cache():
    """Legacy function for backward compatibility"""
    print("Legacy clear function called - use RAGCore.clear_cache() instead")
    return "Cache cleared (legacy function)"

def test_rag_function():
    """Test function to demonstrate RAG retrieval"""

    # Test with new class-based approach
    rag_core = RAGCore()

    # Load vector store
    vector_store = load_data_from_chroma()
    if vector_store is None:
        print("Error: Could not load Chroma vector store.")
        return

    rag_core.set_vector_store(vector_store)

    test_queries = [
        "How do cats use their sense of smell?",
        "What are mutualistic fungi?",
        "Tell me about transportation transformations",
        "Why did people abandon hunting and gathering?",
        "What makes Mercury so dense?",
        "How do marine mammals adapt to their environment?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print(f"{'='*60}")

        result = rag_core.retrieve_documents(query)
        if result['success']:
            print(result['formatted_text'])
        else:
            print(f"Error: {result['error']}")
        print(f"\n{'='*60}")


if __name__ == "__main__":
    # Run test function
    test_rag_function()