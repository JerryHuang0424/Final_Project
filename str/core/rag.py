

def create_rag_prompt(query, context_chunks):
    context_str = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the following context.
If the information is not in the context, say "I don't have enough information in the provided documents to answer that."
Do not make up information or use external knowledge.

Context:
{context_str}

User Question: {query}

Answer:"""
    return prompt