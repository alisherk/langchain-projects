from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever


class FilterRetriever(BaseRetriever):
    db: Chroma
    embeddings: Embeddings

    def _get_relevant_documents(self, query: str):
        query_embedding = self.embeddings.embed_query(query)

        results = self.db.similarity_search_by_vector(query_embedding)

        return results
