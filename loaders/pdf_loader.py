import os
from typing import Any, Dict, List

import fitz


ALLOWED_PDF_FOLDERS = {"policies", "candidate_pack", "evidence"}
IGNORED_FOLDERS = {"expected_outputs", "reference"}


class PDFLoader:
    """Loads PDF documents from the EY dataset folders."""

    def __init__(self, directory: str):
        self.base_directory = self._resolve_directory(directory)

    def _resolve_directory(self, directory: str) -> str:
        candidates = [directory, os.path.join(directory, "ey_dataset")]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    def _iter_allowed_files(self) -> List[tuple[str, str]]:
        file_entries: List[tuple[str, str]] = []

        for entry in sorted(os.listdir(self.base_directory)):
            entry_path = os.path.join(self.base_directory, entry)
            if entry in IGNORED_FOLDERS or entry.lower() == "readme.md":
                print("[SKIP] Ignored file/folder:", entry)
                continue
            if not os.path.isdir(entry_path):
                print("[SKIP] Ignored file/folder:", entry)
                continue
            if entry not in ALLOWED_PDF_FOLDERS:
                print("[SKIP] Ignored file/folder:", entry)
                continue

            for file_name in sorted(os.listdir(entry_path)):
                file_path = os.path.join(entry_path, file_name)
                if os.path.isdir(file_path):
                    print("[SKIP] Ignored file/folder:", file_name)
                    continue
                if file_name.lower().endswith(".pdf"):
                    file_entries.append((entry, file_path))
                else:
                    print("[SKIP] Ignored file/folder:", file_name)

        return file_entries

    def load_single_pdf(self, filepath: str, folder_source: str) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        file_name = os.path.basename(filepath)

        try:
            print("[LOAD] Processing file:", file_name)
            with fitz.open(filepath) as document:
                total_pages = len(document)

                for page_index, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()
                    if len(text) < 20:
                        continue

                    documents.append(
                        {
                            "text": text,
                            "file_name": file_name,
                            "document_type": "pdf",
                            "folder_source": folder_source,
                            "page_number": page_index,
                            "row_index": None,
                            "total_pages": total_pages,
                            "file_path": os.path.abspath(filepath),
                        }
                    )
        except Exception as exc:
            print(f"[LOADER] Failed PDF load: {file_name} ({exc})")

        return documents

    def load_all(self) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for folder_source, filepath in self._iter_allowed_files():
            documents.extend(self.load_single_pdf(filepath, folder_source))
        return documents

    def get_file_list(self) -> List[str]:
        return [os.path.basename(filepath) for _, filepath in self._iter_allowed_files()]
