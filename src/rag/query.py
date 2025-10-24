"""
Query module for VX-RAG.

This module handles user queries, retrieves relevant documents, and formats responses.
"""

from llama_index.core import QueryEngine

def create_query_engine(index):
    """
    Create a query engine from the index.

    Args:
        index: The VectorStoreIndex.

    Returns:
        QueryEngine instance.
    """
    query_engine = index.as_query_engine()
    return query_engine

def query_documents(query: str, query_engine) -> str:
    """
    Query the document corpus.

    Args:
        query: User query string.
        query_engine: The query engine.

    Returns:
        Response string.
    """
    response = query_engine.query(query)
    return str(response)

if __name__ == "__main__":
    # Example usage
    # index = load_index()
    # engine = create_query_engine(index)
    # result = query_documents("What is malware?", engine)
    # print(result)
    pass