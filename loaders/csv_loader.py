import os
from typing import Any, Dict, List, Optional

import pandas as pd


ALLOWED_CSV_FOLDERS = {"structured"}
IGNORED_FOLDERS = {"expected_outputs", "reference"}


class CSVLoader:
    """Loads EY structured CSV files and exposes row helpers."""

    def __init__(self, directory: str):
        self.base_directory = self._resolve_directory(directory)
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._folder_sources: Dict[str, str] = {}
        self._load_all_csvs()

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
            if entry not in ALLOWED_CSV_FOLDERS:
                if entry not in {"policies", "candidate_pack", "evidence"}:
                    print("[SKIP] Ignored file/folder:", entry)
                continue

            for file_name in sorted(os.listdir(entry_path)):
                file_path = os.path.join(entry_path, file_name)
                if os.path.isdir(file_path):
                    print("[SKIP] Ignored file/folder:", file_name)
                    continue
                if file_name.lower().endswith(".csv"):
                    file_entries.append((entry, file_path))
                else:
                    print("[SKIP] Ignored file/folder:", file_name)

        return file_entries

    def _load_all_csvs(self) -> None:
        for folder_source, filepath in self._iter_allowed_files():
            file_name = os.path.basename(filepath)
            try:
                print("[LOAD] Processing file:", file_name)
                dataframe = pd.read_csv(filepath)
                self._dataframes[file_name] = dataframe
                self._folder_sources[file_name] = folder_source
                print(
                    f"[LOADER] CSV: {file_name} "
                    f"({len(dataframe)} rows, {len(dataframe.columns)} columns)"
                )
            except Exception as exc:
                print(f"[LOADER] Failed CSV load: {file_name} ({exc})")

    def get_dataframe(self, file_name: str) -> Optional[pd.DataFrame]:
        if file_name in self._dataframes:
            return self._dataframes[file_name]
        return self._dataframes.get(os.path.basename(file_name))

    def get_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        return self._dataframes

    def get_summary(self, file_name: str) -> Dict[str, Any]:
        dataframe = self.get_dataframe(file_name)
        if dataframe is None:
            return {"error": f"File {file_name} not found"}

        summary: Dict[str, Any] = {
            "file_name": os.path.basename(file_name),
            "folder_source": self._folder_sources.get(os.path.basename(file_name), "structured"),
            "rows": len(dataframe),
            "columns": list(dataframe.columns),
            "missing_values": dataframe.isnull().sum().to_dict(),
            "sample_rows": dataframe.head(3).to_dict(orient="records"),
        }

        numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
        if numeric_columns:
            summary["numeric_stats"] = dataframe[numeric_columns].describe().to_dict()

        return summary

    def filter_rows(
        self,
        file_name: str,
        column: str,
        value: Any,
        operator: str = "eq",
    ) -> pd.DataFrame:
        dataframe = self.get_dataframe(file_name)
        if dataframe is None or column not in dataframe.columns:
            return pd.DataFrame()

        operations = {
            "eq": lambda frame: frame[frame[column] == value],
            "ne": lambda frame: frame[frame[column] != value],
            "gt": lambda frame: frame[frame[column] > value],
            "lt": lambda frame: frame[frame[column] < value],
            "gte": lambda frame: frame[frame[column] >= value],
            "lte": lambda frame: frame[frame[column] <= value],
            "contains": lambda frame: frame[
                frame[column].astype(str).str.contains(str(value), case=False, na=False)
            ],
        }

        operation = operations.get(operator)
        if operation is None:
            return pd.DataFrame()

        return operation(dataframe)

    def search_rows(self, file_name: str, query: str) -> pd.DataFrame:
        dataframe = self.get_dataframe(file_name)
        if dataframe is None or not query.strip():
            return pd.DataFrame()

        query_terms = [term for term in query.lower().split() if len(term) > 2]
        if not query_terms:
            query_terms = [query.lower()]

        text_frame = dataframe.fillna("").astype(str).apply(lambda column: column.str.lower())
        mask = pd.Series(False, index=dataframe.index)

        for term in query_terms:
            term_hits = text_frame.apply(lambda column: column.str.contains(term, na=False))
            mask |= term_hits.any(axis=1)

        return dataframe[mask]

    def get_column_stats(self, file_name: str, column: str) -> Dict[str, Any]:
        dataframe = self.get_dataframe(file_name)
        if dataframe is None or column not in dataframe.columns:
            return {"error": "File or column not found"}

        series = dataframe[column]
        stats: Dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "total_values": len(series),
            "missing_values": int(series.isnull().sum()),
            "unique_values": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            stats.update(
                {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
        else:
            stats["top_values"] = series.astype(str).value_counts().head(10).to_dict()

        return stats

    def rows_to_documents(self) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []

        for file_name, dataframe in self._dataframes.items():
            for row_index, row in dataframe.iterrows():
                parts = []
                for column in dataframe.columns:
                    value = row[column]
                    if pd.isna(value) or str(value).strip() == "":
                        continue
                    parts.append(f"{column}: {value}")

                text = " | ".join(parts)
                if not text:
                    continue

                documents.append(
                    {
                        "text": text,
                        "file_name": file_name,
                        "document_type": "csv",
                        "folder_source": self._folder_sources.get(file_name, "structured"),
                        "page_number": None,
                        "row_index": int(row_index) + 1,
                        "file_path": os.path.join(self.base_directory, self._folder_sources.get(file_name, "structured"), file_name),
                    }
                )

        return documents

    def row_to_record(self, file_name: str, row: pd.Series) -> Dict[str, Any]:
        data = {
            key: (None if pd.isna(value) else value)
            for key, value in row.to_dict().items()
        }
        entity_name = None
        for key in ["full_name", "candidate_name", "employee_name", "department", "audit_id", "candidate_id"]:
            value = data.get(key)
            if value and str(value).strip():
                entity_name = str(value).strip()
                break
        return {
            "file_name": os.path.basename(file_name),
            "document_type": "csv",
            "folder_source": self._folder_sources.get(os.path.basename(file_name), "structured"),
            "page_number": None,
            "row_index": int(row.name) + 1,
            "entity_name": entity_name,
            "record": data,
        }

    def detect_missing_fields(self, file_name: str) -> List[Dict[str, Any]]:
        dataframe = self.get_dataframe(file_name)
        if dataframe is None:
            return []

        missing: List[Dict[str, Any]] = []
        for row_index, row in dataframe.iterrows():
            empty_fields = []
            for column in dataframe.columns:
                value = row[column]
                if pd.isna(value) or str(value).strip() == "":
                    empty_fields.append(column)

            if empty_fields:
                record = self.row_to_record(file_name, row)
                missing.append(
                    {
                        "file_name": os.path.basename(file_name),
                        "document_type": "csv",
                        "folder_source": self._folder_sources.get(os.path.basename(file_name), "structured"),
                        "page_number": None,
                        "row_index": int(row_index) + 1,
                        "entity_name": record.get("entity_name"),
                        "missing_fields": empty_fields,
                    }
                )

        return missing

    def get_file_list(self) -> List[str]:
        return sorted(self._dataframes.keys())

    def get_folder_source(self, file_name: str) -> str:
        return self._folder_sources.get(os.path.basename(file_name), "structured")
