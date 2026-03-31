import streamlit as st
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import os

# 向量存储目录
VECTOR_STORE_DIR = Path("data/vector_store")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

def save_vector_store(vector_store, store_name="default"):
    """保存向量存储到磁盘"""
    try:
        store_path = VECTOR_STORE_DIR / store_name
        vector_store.save_local(str(store_path))
        print(f"Vector store saved to {store_path}")
        return True
    except Exception as e:
        print(f"Error saving vector store: {e}")
        return False

def load_vector_store(embedding_model_instance, store_name="default"):
    """从磁盘加载向量存储"""
    try:
        store_path = VECTOR_STORE_DIR / store_name
        if store_path.exists():
            vector_store = FAISS.load_local(
                str(store_path),
                embedding_model_instance,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {store_path}")
            return vector_store
        else:
            print(f"Vector store not found at {store_path}")
            return None
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def update_vector_store(chunks, embedding_model_instance):
    """更新向量存储，支持持久化"""
    if not chunks:
        st.warning("No chunks to add to the vector store.")
        return st.session_state.vector_store

    if embedding_model_instance is None:
        st.error("Embedding model is not available. Cannot update vector store.")
        return st.session_state.vector_store

    # 创建嵌入模型实例
    EMBEDDING_MODEL_INSTANCE = HuggingFaceEmbeddings(
        model_name=embedding_model_instance
    )

    try:
        # 检查是否已经有向量存储
        if st.session_state.vector_store is None:
            # 尝试从磁盘加载
            st.session_state.vector_store = load_vector_store(EMBEDDING_MODEL_INSTANCE)

            if st.session_state.vector_store is None:
                # 创建新的向量存储
                st.session_state.vector_store = FAISS.from_documents(chunks, EMBEDDING_MODEL_INSTANCE)

                # GPU支持
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    st.session_state.vector_store.index = faiss.index_cpu_to_gpu(res, 0, st.session_state.vector_store.index)
                    print("Using GPU for FAISS")
                else:
                    print("Using CPU for FAISS")
            else:
                # 从磁盘加载成功，添加新文档
                st.session_state.vector_store.add_documents(chunks)
        else:
            # 已有向量存储，直接添加新文档
            st.session_state.vector_store.add_documents(chunks)

        # 保存到磁盘
        save_vector_store(st.session_state.vector_store)

    except Exception as e:
        st.error(f"Error updating vector store: {e}")

    return st.session_state.vector_store

def delete_vector_store(store_name="default"):
    """删除向量存储文件"""
    try:
        store_path = VECTOR_STORE_DIR / store_name
        if store_path.exists():
            # 删除所有相关文件
            for file_path in store_path.glob("*"):
                file_path.unlink()
            store_path.rmdir()
            print(f"Vector store deleted: {store_path}")
            return True
        else:
            print(f"Vector store not found: {store_path}")
            return False
    except Exception as e:
        print(f"Error deleting vector store: {e}")
        return False