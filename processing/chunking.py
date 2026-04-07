"""
Text Chunking — Split documents into smaller chunks for embedding.

Uses LangChain's RecursiveCharacterTextSplitter with metadata preservation.
"""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    """Split documents into overlapping chunks while preserving metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split a list of documents into smaller chunks.

        Each input document dict must have a 'text' key. All other keys
        are treated as metadata and preserved in each output chunk.

        Args:
            documents: List of document dicts with 'text' and metadata keys.

        Returns:
            List of chunk dicts, each containing:
              - text: the chunk text
              - chunk_index: position of this chunk within its parent document
              - all original metadata fields
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            if not text.strip():
                continue

            # Extract metadata (everything except 'text')
            metadata = {k: v for k, v in doc.items() if k != "text"}

            # Split the text
            chunks = self.splitter.split_text(text)

            for i, chunk_text in enumerate(chunks):
                chunk_doc = {
                    "text": chunk_text,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **metadata,
                }
                all_chunks.append(chunk_doc)

        print(f"  Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks

    def chunk_single(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk a single text string.

        Args:
            text: The text to split.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of chunk dicts.
        """
        if metadata is None:
            metadata = {}

        chunks = self.splitter.split_text(text)
        return [
            {"text": chunk, "chunk_index": i, "total_chunks": len(chunks), **metadata}
            for i, chunk in enumerate(chunks)
        ]
