import json
from typing import TYPE_CHECKING, List

from langchain.tools import Tool

if TYPE_CHECKING:
    from processing.embeddings import EmbeddingGenerator
    from vectorstore.faiss_store import FAISSStore


def create_pdf_search_tool(
    faiss_store: "FAISSStore",
    embedding_generator: "EmbeddingGenerator",
) -> Tool:
    """Builds the semantic PDF search tool."""

    def _search_pdfs(query: str) -> str:
        try:
            query_embedding = embedding_generator.embed_query(query)
            matches = faiss_store.search(
                query_embedding=query_embedding,
                top_k=8,
                filter_type="pdf",
            )

            print("[RETRIEVAL] Docs:", len(matches))

            formatted_matches: List[dict] = []
            for match in matches:
                formatted_matches.append(
                    {
                        "file_name": match.get("file_name", "unknown"),
                        "document_type": match.get("document_type", "pdf"),
                        "folder_source": match.get("folder_source"),
                        "page_number": match.get("page_number"),
                        "row_index": match.get("row_index"),
                        "chunk_index": match.get("chunk_index"),
                        "score": round(match.get("score", 0.0), 4),
                        "snippet": match.get("text", "")[:700],
                    }
                )

            return json.dumps(
                {
                    "status": "success" if formatted_matches else "no_results",
                    "query": query,
                    "matches": formatted_matches,
                },
                indent=2,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "status": "error",
                    "query": query,
                    "message": f"PDF search failed: {exc}",
                    "matches": [],
                }
            )

    return Tool(
        name="pdf_search",
        func=_search_pdfs,
        description=(
            "Semantic search across PDF policies. "
            "Use it for rules, limits, exceptions, adjudication criteria, and policy wording."
        ),
    )
