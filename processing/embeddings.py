from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Thin wrapper around sentence-transformers embeddings."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self.model = SentenceTransformer(model)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed_documents(self, documents: List[Dict[str, Any]]) -> tuple:
        texts = [document["text"] for document in documents]
        metadata = [{key: value for key, value in document.items() if key != "text"} for document in documents]
        return self.embed_texts(texts), metadata
