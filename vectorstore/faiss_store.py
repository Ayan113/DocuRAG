import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np


class FAISSStore:
    """Small FAISS wrapper with metadata and duplicate protection."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.document_ids: set[str] = set()

    def _build_document_id(self, text: str, metadata: Dict[str, Any]) -> str:
        identity = {
            "file_name": metadata.get("file_name"),
            "document_type": metadata.get("document_type"),
            "folder_source": metadata.get("folder_source"),
            "page_number": metadata.get("page_number"),
            "row_index": metadata.get("row_index"),
            "chunk_index": metadata.get("chunk_index"),
            "text": text,
        }
        payload = json.dumps(identity, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        if not texts or not embeddings:
            return

        new_texts: List[str] = []
        new_embeddings: List[List[float]] = []
        new_metadata: List[Dict[str, Any]] = []

        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            document_id = self._build_document_id(text, metadata)
            if document_id in self.document_ids:
                continue

            stored_metadata = dict(metadata)
            stored_metadata["document_id"] = document_id
            self.document_ids.add(document_id)
            new_texts.append(text)
            new_embeddings.append(embedding)
            new_metadata.append(stored_metadata)

        if not new_embeddings:
            print("[INDEX] No new documents to add")
            return

        vectors = np.array(new_embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.texts.extend(new_texts)
        self.metadata.extend(new_metadata)

        print(f"[INDEX] Added {len(new_embeddings)} vectors (total: {self.index.ntotal})")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        search_k = min(self.index.ntotal, top_k * 4 if filter_type else top_k)
        scores, indices = self.index.search(query_vector, search_k)

        results: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            metadata = self.metadata[index]
            if filter_type and metadata.get("document_type") != filter_type:
                continue

            results.append(
                {
                    "text": self.texts[index],
                    "score": float(score),
                    **metadata,
                }
            )
            if len(results) >= top_k:
                break

        return results

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metadata": self.metadata,
                    "texts": self.texts,
                    "dimension": self.dimension,
                },
                handle,
                indent=2,
            )

        print(f"[INDEX] Saved index to {directory}")

    def load(self, directory: str) -> None:
        index_path = os.path.join(directory, "faiss.index")
        metadata_path = os.path.join(directory, "metadata.json")
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"FAISS index files not found in {directory}")

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.metadata = data.get("metadata", [])
        self.texts = data.get("texts", [])
        self.dimension = data.get("dimension", self.dimension)
        self.document_ids = {
            item.get("document_id")
            for item in self.metadata
            if item.get("document_id")
        }

        print(f"[INDEX] Loaded index from {directory} ({self.index.ntotal} vectors)")

    def get_stats(self) -> Dict[str, Any]:
        type_distribution: Dict[str, int] = {}
        source_distribution: Dict[str, int] = {}
        folder_distribution: Dict[str, int] = {}

        for item in self.metadata:
            document_type = item.get("document_type", "unknown")
            file_name = item.get("file_name", "unknown")
            folder_source = item.get("folder_source", "unknown")
            type_distribution[document_type] = type_distribution.get(document_type, 0) + 1
            source_distribution[file_name] = source_distribution.get(file_name, 0) + 1
            folder_distribution[folder_source] = folder_distribution.get(folder_source, 0) + 1

        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "type_distribution": type_distribution,
            "source_distribution": source_distribution,
            "folder_distribution": folder_distribution,
        }
